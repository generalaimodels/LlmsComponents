import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # Make SwiGLU hidden layer size a multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(nn.Module):
    """RMSNorm layer for normalizing inputs."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize the RMSNorm layer.

        Args:
            dim (int): Dimension of the input tensor.
            eps (float): Epsilon value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass."""
        output = self._norm(x.float()).type_as(x)
        return self.weight * output


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute frequencies for rotary embeddings.

    Args:
        dim (int): Dimension of the embeddings.
        end (int): Maximum sequence length.
        theta (float): Base frequency.

    Returns:
        torch.Tensor: Precomputed frequencies tensor.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # Outer product
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Complex64 tensor
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape freqs_cis tensor for broadcasting with x.

    Args:
        freqs_cis (torch.Tensor): Tensor of precomputed frequencies.
        x (torch.Tensor): Tensor to be broadcasted with.

    Returns:
        torch.Tensor: Reshaped freqs_cis tensor.
    """
    ndim = x.ndim
    shape = [1] * ndim
    shape[1] = x.shape[1]
    shape[-1] = x.shape[-1]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding to query and key tensors.

    Args:
        xq (torch.Tensor): Query tensor.
        xk (torch.Tensor): Key tensor.
        freqs_cis (torch.Tensor): Precomputed rotations.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(start_dim=-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(start_dim=-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat the key/value tensors along the head dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_kv_heads, head_dim).
        n_rep (int): Number of repetitions.

    Returns:
        torch.Tensor: Tensor with repeated heads.
    """
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x.unsqueeze(3)  # Shape: (batch_size, seq_len, n_kv_heads, 1, head_dim)
    x = x.expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
    x = x.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    return x


class Attention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(self, args: ModelArgs) -> None:
        """Initialize the Attention layer.

        Args:
            args (ModelArgs): Model arguments.
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads else self.n_heads
        self.head_dim = args.dim // self.n_heads

        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = None
        self.cache_v = None
        self.max_seq_len = args.max_seq_len
        self.max_batch_size = args.max_batch_size

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Perform forward pass for the Attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Start position in the sequence.
            freqs_cis (torch.Tensor): Precomputed frequencies.
            mask (Optional[torch.Tensor]): Attention mask.

        Returns:
            torch.Tensor: Output tensor after attention.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        if self.cache_k is None or self.cache_v is None:
            cache_shape = (
                batch_size,
                self.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
            self.cache_k = torch.zeros(cache_shape, device=device, dtype=xk.dtype)
            self.cache_v = torch.zeros(cache_shape, device=device, dtype=xv.dtype)

        # Update key and value caches
        self.cache_k[:batch_size, start_pos : start_pos + seq_len, :, :] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len, :, :] = xv

        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Repeat keys and values if n_kv_heads < n_heads
        if self.n_kv_heads < self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            keys = repeat_kv(keys, n_rep)
            values = repeat_kv(values, n_rep)

        xq = xq.transpose(1, 2)  # Shape: (batch_size, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # Shape: (batch_size, n_heads, total_seq_len, head_dim)
        values = values.transpose(1, 2)  # Same shape as keys

        # Compute attention scores
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores += mask

        # Apply softmax to get attention probabilities
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # Shape: (batch_size, n_heads, seq_len, head_dim)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """FeedForward network with optional SwiGLU activation."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ) -> None:
        """Initialize the FeedForward layer.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            multiple_of (int): Multiple for hidden dimension.
            ffn_dim_multiplier (Optional[float]): Multiplier for hidden dimension.
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Transformer block consisting of attention and feedforward layers."""

    def __init__(self, layer_id: int, args: ModelArgs) -> None:
        """Initialize the TransformerBlock.

        Args:
            layer_id (int): Layer index.
            args (ModelArgs): Model arguments.
        """
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Perform forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Start position in the sequence.
            freqs_cis (torch.Tensor): Precomputed frequencies for rotary embedding.
            mask (Optional[torch.Tensor]): Attention mask.

        Returns:
            torch.Tensor: Output tensor.
        """
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """Transformer model combining multiple Transformer blocks."""

    def __init__(self, params: ModelArgs) -> None:
        """Initialize the Transformer model.

        Args:
            params (ModelArgs): Model arguments.
        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = nn.ModuleList(
            [TransformerBlock(layer_id, params) for layer_id in range(self.n_layers)]
        )

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Perform forward pass for the Transformer.

        Args:
            tokens (torch.Tensor): Input tokens of shape (batch_size, seq_len).
            start_pos (int): Start position in the sequence.

        Returns:
            torch.Tensor: Output logits.
        """
        batch_size, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen].to(h.device)

        # Create attention mask if needed
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            if start_pos > 0:
                prefix = torch.zeros((seqlen, start_pos), device=h.device)
                mask = torch.cat((prefix, mask), dim=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seqlen, total_seq_len)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h)
        return output.float()

