from typing import Any, Callable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class GeneralizedGatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dims: List[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts

        if hidden_dims is None:
            self.network = nn.Linear(input_dim, num_experts)
        else:
            layers = []
            prev_dim = input_dim
            for dim in hidden_dims:
                layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
                prev_dim = dim
            layers.append(nn.Linear(prev_dim, num_experts))
            self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class Top2Gating(nn.Module):
    def __init__(self, model_dim: int, num_experts: int):
        super().__init__()
        self.wg = nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.wg(input)
        return top2gating(logits)

def top2gating(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gates = F.softmax(logits, dim=1, dtype=torch.float)
    
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    capacity = 2 * num_tokens // num_experts
    
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
    
    logits_except1 = logits.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)
    
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    locations2 += torch.sum(mask1, dim=0, keepdim=True)
    
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce)
    
    mask1 *= (locations1 < capacity)
    mask2 *= (locations2 < capacity)
    
    gates1_s = (gates * mask1).sum(dim=1)
    gates2_s = (gates * mask2).sum(dim=1)
    denom_s = torch.clamp(gates1_s + gates2_s, min=torch.finfo(gates1_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s
    
    locations1_s = (locations1 * mask1).sum(dim=1)
    locations2_s = (locations2 * mask2).sum(dim=1)
    
    combine1_sec = gates1_s.unsqueeze(-1).unsqueeze(-1) * F.one_hot(locations1_s, num_classes=capacity).unsqueeze(1)
    combine2_sec = gates2_s.unsqueeze(-1).unsqueeze(-1) * F.one_hot(locations2_s, num_classes=capacity).unsqueeze(1)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()
    
    return l_aux.to(logits.dtype), combine_weights.to(logits.dtype), dispatch_mask

class GeneralizedMoELayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        expert_fn: Callable[[int, int], nn.Module],
        gating_fn: Callable[[int, int], nn.Module] = Top2Gating,
        group: Optional[Any] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.gate = gating_fn(input_dim, num_experts)
        self.experts = nn.ModuleList([expert_fn(input_dim, output_dim) for _ in range(num_experts)])
        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)
        self.num_local_experts = len(self.experts)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = input.shape
        reshaped_input = input.reshape(-1, self.input_dim)
        
        l_aux, combine_weights, dispatch_mask = self.gate(reshaped_input)
        
        dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.float(), reshaped_input)
        dispatched_input = self._all_to_all(dispatched_input)
        dispatched_input = dispatched_input.reshape(self.world_size, self.num_local_experts, -1, self.input_dim)
        
        chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        expert_outputs = [expert(chunk) for chunk, expert in zip(chunks, self.experts)]
        expert_output = torch.cat(expert_outputs, dim=1)
        
        expert_output = self._all_to_all(expert_output)
        expert_output = expert_output.reshape(self.world_size * self.num_local_experts, -1, self.output_dim)
        
        combined_output = torch.einsum("sec,ecm->sm", combine_weights, expert_output)
        output = combined_output.reshape(original_shape[:-1] + (self.output_dim,))
        
        return output, l_aux

    def _all_to_all(self, tensor: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(tensor)
        dist.all_to_all_single(output, tensor, group=self.group)
        return output

def create_mlp_expert(input_dim: int, output_dim: int, hidden_dims: List[int] = None) -> nn.Module:
    if hidden_dims is None:
        return nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())

    layers = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

class DenseMoEModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        expert_hidden_dims: List[int],
        num_moe_layers: int,
        gating_fn: Callable[[int, int], nn.Module] = Top2Gating,
        group: Optional[Any] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.num_moe_layers = num_moe_layers

        def expert_fn(in_dim: int, out_dim: int) -> nn.Module:
            return create_mlp_expert(in_dim, out_dim, expert_hidden_dims)

        self.moe_layers = nn.ModuleList([
            GeneralizedMoELayer(
                input_dim,
                input_dim,  # Keep the dimension same for intermediate layers
                num_experts,
                expert_fn,
                gating_fn,
                group
            ) for _ in range(num_moe_layers)
        ])

        self.final_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total_aux_loss = 0.0
        for moe_layer in self.moe_layers:
            x, l_aux = moe_layer(x)
            total_aux_loss += l_aux

        output = self.final_layer(x)
        return output, total_aux_loss


def example_usage():
    input_dim = 512
    output_dim = 256
    num_experts = 8
    expert_hidden_dims = [1024, 1024]
    num_moe_layers = 4
    batch_size = 32
    seq_length = 128

    model = DenseMoEModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        expert_hidden_dims=expert_hidden_dims,
        num_moe_layers=num_moe_layers,
    )

    # Generate random input
    x = torch.randn(batch_size, seq_length, input_dim)

    # Forward pass
    output, aux_loss = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {aux_loss.item()}")


if __name__ == "__main__":
    example_usage()