"""Minimal decoder-only Transformer with per-layer execution."""

import torch
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    max_seq_len: int = 512


class Attention:
    """Multi-head causal self-attention."""

    def __init__(self, config: TransformerConfig):
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.hidden_dim = config.hidden_dim

        # Parameters stored as plain tensors (CPU by default)
        self.wq = torch.randn(config.hidden_dim, config.hidden_dim) * 0.02
        self.wk = torch.randn(config.hidden_dim, config.hidden_dim) * 0.02
        self.wv = torch.randn(config.hidden_dim, config.hidden_dim) * 0.02
        self.wo = torch.randn(config.hidden_dim, config.hidden_dim) * 0.02

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (batch, seq, hidden)"""
        B, S, _ = x.shape

        q = (x @ self.wq).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = (x @ self.wk).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = (x @ self.wv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is None:
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, self.hidden_dim)
        return out @ self.wo

    def to(self, device: torch.device):
        self.wq = self.wq.to(device)
        self.wk = self.wk.to(device)
        self.wv = self.wv.to(device)
        self.wo = self.wo.to(device)
        return self

    def parameters(self):
        return [self.wq, self.wk, self.wv, self.wo]


class MLP:
    """Feed-forward network."""

    def __init__(self, config: TransformerConfig):
        self.hidden_dim = config.hidden_dim
        self.intermediate = config.hidden_dim * 4

        self.w1 = torch.randn(config.hidden_dim, self.intermediate) * 0.02
        self.w2 = torch.randn(self.intermediate, config.hidden_dim) * 0.02

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x @ self.w1) @ self.w2

    def to(self, device: torch.device):
        self.w1 = self.w1.to(device)
        self.w2 = self.w2.to(device)
        return self

    def parameters(self):
        return [self.w1, self.w2]


class RMSNorm:
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = torch.ones(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

    def to(self, device: torch.device):
        self.weight = self.weight.to(device)
        return self

    def parameters(self):
        return [self.weight]


class TransformerLayer:
    """Single transformer decoder layer."""

    def __init__(self, config: TransformerConfig):
        self.attn = Attention(config)
        self.mlp = MLP(config)
        self.norm1 = RMSNorm(config.hidden_dim)
        self.norm2 = RMSNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn.forward(self.norm1.forward(x), mask)
        x = x + self.mlp.forward(self.norm2.forward(x))
        return x

    def to(self, device: torch.device):
        self.attn.to(device)
        self.mlp.to(device)
        self.norm1.to(device)
        self.norm2.to(device)
        return self

    def parameters(self):
        return self.attn.parameters() + self.mlp.parameters() + self.norm1.parameters() + self.norm2.parameters()


class Embedding:
    """Token + position embedding."""

    def __init__(self, config: TransformerConfig):
        self.tok_emb = torch.randn(config.vocab_size, config.hidden_dim) * 0.02
        self.pos_emb = torch.randn(config.max_seq_len, config.hidden_dim) * 0.02

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (batch, seq)"""
        seq_len = token_ids.shape[1]
        return self.tok_emb[token_ids] + self.pos_emb[:seq_len]

    def to(self, device: torch.device):
        self.tok_emb = self.tok_emb.to(device)
        self.pos_emb = self.pos_emb.to(device)
        return self

    def parameters(self):
        return [self.tok_emb, self.pos_emb]


class OutputHead:
    """Final norm + projection to vocab."""

    def __init__(self, config: TransformerConfig):
        self.norm = RMSNorm(config.hidden_dim)
        self.proj = torch.randn(config.hidden_dim, config.vocab_size) * 0.02

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm.forward(x) @ self.proj

    def to(self, device: torch.device):
        self.norm.to(device)
        self.proj = self.proj.to(device)
        return self

    def parameters(self):
        return self.norm.parameters() + [self.proj]
