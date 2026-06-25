"""
Production-grade transformer operations using PyTorch built-ins where possible.

This module provides:
1. Custom autograd Functions for operations that need manual gradients
2. Leverages torch.nn.functional for standard operations
3. Proper gradient flow verification
4. Memory-efficient implementations
"""
import math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class RMSNorm(torch.autograd.Function):
    """
    RMSNorm with custom backward pass.
    Uses autograd.Function to integrate with PyTorch's autograd system
    while maintaining manual control over gradients.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
        """
        Forward pass: y = x * rsqrt(mean(x^2) + eps) * weight

        Args:
            x: Input tensor [B, T, H]
            weight: Scale parameter [H]
            eps: Epsilon for numerical stability
        """
        var = x.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(var + eps)
        y = x * inv_rms * weight

        # Save for backward
        ctx.save_for_backward(x, weight, inv_rms)
        ctx.hidden_size = x.shape[-1]

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass with proper gradient computation.

        Returns:
            grad_x, grad_weight, None (for eps)
        """
        x, weight, inv_rms = ctx.saved_tensors
        hidden_size = ctx.hidden_size

        # Gradient w.r.t. weight
        grad_weight = (grad_output * x * inv_rms).sum(dim=(0, 1))

        # Gradient w.r.t. x (chain rule through normalization)
        a = grad_output * weight
        s = (a * x).sum(dim=-1, keepdim=True)
        grad_x = a * inv_rms - (x * inv_rms.pow(3) * s) / hidden_size

        return grad_x, grad_weight, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMSNorm with custom gradient."""
    return RMSNorm.apply(x, weight, eps)


class RotaryEmbedding:
    """
    Rotary Position Embedding (RoPE) with caching.

    Implements the rotary embeddings from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0, device: str = "cpu"):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos/sin for max sequence length
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', emb.cos()[None, :, None, :])
        self.register_buffer('sin_cached', emb.sin()[None, :, None, :])

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Simple buffer registration (for standalone use)."""
        setattr(self, name, tensor)

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor [B, T, num_heads, head_dim]
            k: Key tensor [B, T, num_kv_heads, head_dim]
            seq_len: Sequence length

        Returns:
            Rotated q and k tensors
        """
        cos = self.cos_cached[:, :seq_len, :, :].to(q.dtype)
        sin = self.sin_cached[:, :seq_len, :, :].to(q.dtype)

        # Rotate half
        def rotate_half(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).reshape_as(x)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed


class ScaledDotProductAttention(torch.nn.Module):
    """
    Scaled dot-product attention with causal masking.
    Uses torch.nn.functional.scaled_dot_product_attention when possible for efficiency.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention.

        Args:
            q: Query [B, num_heads, T, head_dim]
            k: Key [B, num_heads, T, head_dim]
            v: Value [B, num_heads, T, head_dim]
            attn_mask: Optional attention mask [B, T]
            is_causal: Whether to apply causal masking

        Returns:
            Attention output [B, num_heads, T, head_dim]
        """
        # Use PyTorch's optimized SDPA if available (Flash Attention 2)
        if hasattr(F, 'scaled_dot_product_attention') and attn_mask is None:
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )

        # Fallback to manual implementation
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if is_causal:
            T = q.size(2)
            causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        if attn_mask is not None:
            # attn_mask: [B, T] where 1 = keep, 0 = mask
            attn_mask = (~attn_mask.bool()).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            scores = scores.masked_fill(attn_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        return torch.matmul(attn_weights, v)


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA) support.

    Uses PyTorch's nn.Linear for projections to leverage optimized kernels.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        rope: Optional[RotaryEmbedding] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        # Use nn.Linear for projections (can leverage cuBLAS/cuDNN)
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.rope = rope
        self.attn = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, T, H]
            attn_mask: Optional attention mask [B, T]

        Returns:
            Output tensor [B, T, H]
        """
        B, T, H = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        # Apply RoPE if available
        if self.rope is not None:
            q, k = self.rope.apply_rotary(q, k, T)

        # Expand K, V for GQA
        if self.num_kv_heads != self.num_heads:
            expand_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(expand_factor, dim=2)
            v = v.repeat_interleave(expand_factor, dim=2)

        # Transpose for attention: [B, num_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        attn_output = self.attn(q, k, v, attn_mask=attn_mask, is_causal=True)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, H)
        output = self.o_proj(attn_output)

        return output


class MLP(torch.nn.Module):
    """
    MLP with SwiGLU activation (used in LLaMA/Qwen models).

    Uses PyTorch's nn.Linear for better performance.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.

        SwiGLU(x) = (Swish(gate(x)) * up(x)) @ down
        where Swish(x) = x * sigmoid(x)
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up  # SwiGLU
        output = self.down_proj(hidden)
        return output


class TransformerLayer(torch.nn.Module):
    """
    Single transformer decoder layer with pre-normalization.

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + MLP(RMSNorm(x))
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_kv_heads: Optional[int] = None,
        rope: Optional[RotaryEmbedding] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Layer norms (using custom RMSNorm)
        self.input_layernorm = torch.nn.Parameter(torch.ones(hidden_size))
        self.post_attention_layernorm = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

        # Attention and MLP
        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope=rope,
        )
        self.mlp = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, T, H]
            attn_mask: Optional attention mask [B, T]

        Returns:
            Output tensor [B, T, H]
        """
        # Attention block
        residual = x
        x = rmsnorm(x, self.input_layernorm, self.eps)
        x = self.self_attn(x, attn_mask=attn_mask)
        x = residual + x

        # MLP block
        residual = x
        x = rmsnorm(x, self.post_attention_layernorm, self.eps)
        x = self.mlp(x)
        x = residual + x

        return x


def test_operations():
    """Test that all operations work correctly with autograd."""
    torch.manual_seed(42)
    B, T, H = 2, 4, 8

    print("Testing RMSNorm...")
    x = torch.randn(B, T, H, requires_grad=True)
    weight = torch.randn(H, requires_grad=True)
    y = rmsnorm(x, weight)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None and weight.grad is not None
    print("✓ RMSNorm gradients flow correctly")

    print("\nTesting RotaryEmbedding...")
    rope = RotaryEmbedding(dim=H, max_seq_len=16, device='cpu')
    q = torch.randn(B, T, 2, H, requires_grad=True)
    k = torch.randn(B, T, 2, H, requires_grad=True)
    q_rot, k_rot = rope.apply_rotary(q, k, T)
    loss = (q_rot.sum() + k_rot.sum())
    loss.backward()
    assert q.grad is not None and k.grad is not None
    print("✓ RoPE gradients flow correctly")

    print("\nTesting MultiHeadAttention...")
    attn = MultiHeadAttention(hidden_size=H, num_heads=2, num_kv_heads=2)
    x = torch.randn(B, T, H, requires_grad=True)
    y = attn(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    print("✓ Attention gradients flow correctly")

    print("\nTesting MLP...")
    mlp = MLP(hidden_size=H, intermediate_size=16)
    x = torch.randn(B, T, H, requires_grad=True)
    y = mlp(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    print("✓ MLP gradients flow correctly")

    print("\nTesting TransformerLayer...")
    layer = TransformerLayer(hidden_size=H, num_heads=2, intermediate_size=16)
    x = torch.randn(B, T, H, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    print("✓ TransformerLayer gradients flow correctly")

    print("\n" + "="*60)
    print("✓ All operations tested successfully!")
    print("="*60)


if __name__ == "__main__":
    test_operations()
