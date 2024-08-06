import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequencies for rotary embeddings.

    Args:
        dim: Dimension of the embeddings.
        max_seq_len: Maximum sequence length.
        theta: Base for the frequency calculation.

    Returns:
        A tensor of precomputed frequencies.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        xq: Query tensor.
        xk: Key tensor.
        freqs_cis: Precomputed frequencies.

    Returns:
        A tuple containing the rotated query and key tensors.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    seq_len = xq_.shape[-2]
    freqs_cis = freqs_cis[:seq_len, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis.transpose(0, 1).unsqueeze(1)).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis.transpose(0, 1).unsqueeze(1)).flatten(3)

    # xq_out = torch.view_as_real(xq_ * freqs_cis.unsqueeze(1)).flatten(3)
    # xk_out = torch.view_as_real(xk_ * freqs_cis.unsqueeze(1)).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    """
    RMS Normalization layer.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Attention(nn.Module):
    """
    Multi-head attention layer.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class FeedForward(nn.Module):
    """
    Feedforward network with GELU activation and dropout.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer.
    """

    def __init__(self, dim: int, num_experts: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            [FeedForward(dim, hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)
        weights = F.softmax(gate_logits, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        output = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=-1)
        return output


class TransformerBlock(nn.Module):
    """
    Transformer block consisting of attention and feedforward layers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = RMSNorm(dim)
        self.ffn = MixtureOfExperts(dim, num_experts=4, hidden_dim=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.ffn(self.norm2(x))
        return x


class ImageGeneratorEncoder(nn.Module):
    """
    Image generator encoder based on transformer architecture.
    """

    def __init__(
        self, img_size: int, patch_size: int, dim: int, depth: int, num_heads: int
    ):
        super().__init__()
        num_patches = img_size // patch_size  # Correct calculation
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.patch_embed = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads) for _ in range(depth)]
        )
        self.norm = RMSNorm(dim)
        self.freqs_cis = precompute_freqs_cis(dim // num_heads, num_patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x, self.freqs_cis)
        x = self.norm(x)
        return x


class ImageGeneratorDecoder(nn.Module):
    """
    Image generator decoder based on transformer architecture.
    """

    def __init__(
        self, img_size: int, patch_size: int, dim: int, depth: int, num_heads: int
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # num_patches = (img_size // patch_size) ** 2
       
        num_patches = img_size // patch_size  # Correct calculation
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
    
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads) for _ in range(depth)]
        )
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, patch_size * patch_size * 3)

        self.freqs_cis = precompute_freqs_cis(dim // num_heads, num_patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x, self.freqs_cis)
        x = self.norm(x)
        x = self.head(x)
        x = x.reshape(-1, self.img_size, self.img_size, 3).permute(0, 3, 1, 2)
        return x


class ImageGenerator(nn.Module):
    """
    Image generator combining encoder and decoder.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        num_heads: int,
        latent_dim: int,
    ):
        super().__init__()
        self.encoder = ImageGeneratorEncoder(
            img_size, patch_size, dim, depth, num_heads
        )
        self.decoder = ImageGeneratorDecoder(
            img_size, patch_size, dim, depth, num_heads
        )
        self.latent_proj = nn.Linear(latent_dim, (img_size // patch_size) ** 2 * dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        latent = self.latent_proj(z).view(z.shape[0], -1, encoded.shape[-1])
        fused = torch.cat([encoded, latent], dim=1)
        generated = self.decoder(fused)
        return generated


class ResidualBlock(nn.Module):
    """
    Residual block with RMSNorm and ReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.norm1 = RMSNorm(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.norm2 = RMSNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        residual = self.shortcut(residual)
        out += residual
        out = self.relu(out)

        return out


class EnhancedImageGenerator(nn.Module):
    """
    Enhanced image generator with residual blocks.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        num_heads: int,
        latent_dim: int,
        num_residual_blocks: int = 3,
    ):
        super().__init__()
        self.transformer_generator = ImageGenerator(
            img_size, patch_size, dim, depth, num_heads, latent_dim
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(3, 64) for _ in range(num_residual_blocks)]
        )

        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        transformer_output = self.transformer_generator(x, z)
        enhanced = self.residual_blocks(transformer_output)
        final_output = self.final_conv(enhanced)
        return torch.tanh(final_output)


# Example usage:
if __name__ == "__main__":
    img_size = 256
    patch_size = 16
    dim = 1024
    depth = 12
    num_heads = 16
    latent_dim = 512

    generator = EnhancedImageGenerator(
        img_size, patch_size, dim, depth, num_heads, latent_dim
    )

    # Generate a batch of 8 images
    batch_size = 8
    x = torch.randn(batch_size, 3, img_size, img_size)
    z = torch.randn(batch_size, latent_dim)

    generated_images = generator(x, z)
    print(generated_images.shape)  # Should output: torch.Size([8, 3, 256, 256])