from typing import List, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedGatingNetwork(nn.Module):
    """A generalized gating network for Dense MoE."""

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
        """Compute gating weights for experts."""
        gating_logits = self.network(x)
        return F.softmax(gating_logits, dim=-1)


class GeneralizedDenseMoE(nn.Module):
    """A generalized Dense Mixture of Experts layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        expert_fn: Callable[[int, int], nn.Module],
        gating_hidden_dims: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts

        self.gating_network = GeneralizedGatingNetwork(input_dim, num_experts, gating_hidden_dims)
        self.experts = nn.ModuleList([expert_fn(input_dim, output_dim) for _ in range(num_experts)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Dense MoE layer."""
        gating_weights = self.gating_network(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return self.dropout(output)


class EfficientDenseMoE(nn.Module):
    """An efficient implementation of Dense Mixture of Experts layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        expert_fn: Callable[[int, int], nn.Module],
        gating_hidden_dims: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts

        self.gating_network = GeneralizedGatingNetwork(input_dim, num_experts, gating_hidden_dims)
        self.experts = expert_fn(input_dim, output_dim * num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Efficient forward pass through the Dense MoE layer."""
        gating_weights = self.gating_network(x)
        expert_outputs = self.experts(x).view(-1, self.num_experts, self.output_dim)
        output = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return self.dropout(output)


def create_mlp_expert(input_dim: int, output_dim: int, hidden_dims: List[int] = None) -> nn.Module:
    """Create a Multi-Layer Perceptron (MLP) expert."""
    if hidden_dims is None:
        return nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())

    layers = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def example_usage():
    """Example usage of the generalized Dense MoE components."""
    input_dim = 128
    output_dim = 64
    num_experts = 4
    batch_size = 32

    # Create an MLP expert function
    def mlp_expert_fn(in_dim: int, out_dim: int) -> nn.Module:
        return create_mlp_expert(in_dim, out_dim, hidden_dims=[256, 128])

    # Create a GeneralizedDenseMoE model
    model = GeneralizedDenseMoE(
        input_dim=input_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        expert_fn=mlp_expert_fn,
        gating_hidden_dims=[64, 32],
        dropout=0.1,
    )

    # Create an EfficientDenseMoE model
    efficient_model = EfficientDenseMoE(
        input_dim=input_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        expert_fn=mlp_expert_fn,
        gating_hidden_dims=[64, 32],
        dropout=0.1,
    )

    # Generate random input
    x = torch.randn(batch_size, input_dim)

    # Forward pass through both models
    output = model(x)
    efficient_output = efficient_model(x)

    print(f"GeneralizedDenseMoE output shape: {output.shape}")
    print(f"EfficientDenseMoE output shape: {efficient_output.shape}")


if __name__ == "__main__":
    example_usage()