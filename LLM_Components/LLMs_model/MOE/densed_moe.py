import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Any, Callable, Optional, Tuple, Union, cast, TYPE_CHECKING
import torch.multiprocessing as mp
from torch.distributed import init_process_group
if TYPE_CHECKING:
    Base = nn.Module[Tensor]
else:
    Base = nn.Module
def init_distributed():
    init_process_group(backend="gloo")  # or use "nccl" if you're using GPUs
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


class Top2Gate(nn.Module):
    def __init__(self, model_dim: int, num_experts: int) -> None:
        super().__init__()
        self.wg = nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, input: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        logits = self.wg(input)
        return top2gating(logits)


def gumbel_rsample(shape: Tuple[int, ...], device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


def one_hot(tensor: torch.Tensor, num_classes: int) -> Tensor:
    assert num_classes > 0, "num_classes must be a positive integer"
    ret = torch.zeros(tensor.shape + (num_classes,), device=tensor.device, dtype=tensor.dtype)
    ret.scatter_(-1, tensor.unsqueeze(-1), 1)
    return ret


def top2gating(logits: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    gates = F.softmax(logits, dim=1, dtype=torch.float)
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    capacity = 2 * num_tokens // num_experts
    assert num_tokens % num_experts == 0

    indices1_s = torch.argmax(gates, dim=1)
    mask1 = one_hot(indices1_s, num_experts)

    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = one_hot(indices2_s, num_experts)

    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce)

    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    gates1_s = (gates * mask1).sum(dim=1)
    gates2_s = (gates * mask2).sum(dim=1)
    denom_s = gates1_s + gates2_s
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    gates1 = gates1_s.unsqueeze(-1) * mask1
    gates2 = gates2_s.unsqueeze(-1) * mask2
    locations1_sc = one_hot(locations1_s, capacity)
    locations2_sc = one_hot(locations2_s, capacity)
    combine1_sec = gates1.unsqueeze(2) * locations1_sc.unsqueeze(1)
    combine2_sec = gates2.unsqueeze(2) * locations2_sc.unsqueeze(1)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux.to(logits.dtype), combine_weights.to(logits.dtype), dispatch_mask


class MOELayer(Base):
    def __init__(self, gate: nn.Module, experts: Union[nn.Module, nn.ModuleList], group: Optional[Any] = None) -> None:
        super().__init__()
        self.gate = gate
        if isinstance(experts, nn.ModuleList):
            self.experts = cast(nn.ModuleList, experts)
        else:
            self.experts = nn.ModuleList([experts])
        self.group = group if group is not None else dist.group.WORLD
        for expert in self.experts:
            for p in experts.parameters():
                p.expert = True  # type: ignore
        self.world_size = dist.get_world_size(self.group)
        self.num_local_experts = len(self.experts)

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        assert len(input) == 1, "only single input Tensor supported"
        assert len(input[0].shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        assert input[0].shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

        d_model = input[0].shape[2]
        reshaped_input = input[0].reshape(-1, d_model)
        self.l_aux, combine_weights, dispatch_mask = self.gate(reshaped_input)
        dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.float(), reshaped_input)
        dispatched_input = _AllToAll.apply(self.group, dispatched_input)
        dispatched_input = dispatched_input.reshape(self.world_size, self.num_local_experts, -1, d_model)
        chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        expert_outputs = [expert(chunk) for chunk, expert in zip(chunks, self.experts)]
        expert_output = torch.cat(expert_outputs, dim=1)
        expert_output = _AllToAll.apply(self.group, expert_output)
        expert_output = expert_output.reshape(self.world_size * self.num_local_experts, -1, d_model)
        combined_output = torch.einsum("sec,ecm->sm", combine_weights, expert_output)
        return combined_output.reshape(input[0].shape)


class GeneralizedDenseMoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        expert_fn: Callable[[int, int], nn.Module]
    ) -> None:
        super(GeneralizedDenseMoE, self).__init__()
        self.gate = Top2Gate(input_dim, num_experts)
        self.experts = nn.ModuleList([expert_fn(input_dim, output_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MOELayer(self.gate, self.experts)(x)


# Example usage
def example_expert_fn(input_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU()
    )

def run_model(rank, world_size):
    init_distributed()
    
    input_dim = 128
    output_dim = 64
    num_experts = 4

    model = GeneralizedDenseMoE(input_dim, output_dim, num_experts, example_expert_fn)
    x = torch.randn(32, input_dim)
    output = model(x)
    print(output.shape)  # Should be (32, output_dim)

if __name__ == "__main__":
    world_size = 1  # Change this to the number of processes you want to use
    mp.spawn(run_model, args=(world_size,), nprocs=world_size, join=True)