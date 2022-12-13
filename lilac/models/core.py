"""
core.py

Core modeling utilities, handy Transformer Module stubs, and other model/inference-related functionality.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


# === Transformer Utilities ===
def get_1D_sine_cosine(dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(dim // 2, dtype=np.float32) / (dim / 2.0)
    omega = 1.0 / (10000**omega)

    out = np.einsum("m,d->md", pos.reshape(-1), omega)  # [flatten(pos) x omega] -- outer product!
    emb_sin, emb_cos = np.sin(out), np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # [flatten(pos) x D]


# 1D Sine-Cosine Position Embedding -- standard from "Attention is all you need!"
def get_1D_position_embeddings(embed_dim: int, length: int) -> np.ndarray:
    return get_1D_sine_cosine(embed_dim, np.arange(length))


# SwishGLU -- A Gated Linear Unit with the Swish Activation; always better than standard Linear in Feed-Forwards!
class SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


# Standard Multi-Headed Self-Attention (w/ masking)
class Attention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        assert embed_dim % n_heads == 0, "`embed_dim` must be divisible by `n_heads`!"
        self.n_heads, self.scale = n_heads, (embed_dim // n_heads) ** -0.5

        # Projections -- Note that some folks recommend dropping QKV bias at scale... but we use it!
        self.qkv, self.proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True), nn.Linear(embed_dim, embed_dim)

    def forward(self, seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project to queries, keys, and values
        q, k, v = [
            rearrange(x, "bsz seq (heads d) -> bsz heads seq d", heads=self.n_heads)
            for x in self.qkv(seq).chunk(3, dim=-1)
        ]

        # Attention -- with masking!
        scores = q @ (k.transpose(-2, -1) * self.scale)
        if mask is not None:
            assert mask.ndim == 2, "Mask should be of dimensionality [bsz, seq]!"
            mask = rearrange(mask, "bsz seq -> bsz 1 seq 1")

            # Mask out by filling indices with negative infinity...
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        # Compute weighted attention sum & return
        attn = scores.softmax(dim=-1)
        return rearrange(attn @ v, "bsz heads seq d -> bsz seq (heads d)")


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.embed_dim, self.n_heads, self.mlp_ratio = embed_dim, n_heads, mlp_ratio

        # Create Block Components
        self.pre_attn_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.attn = Attention(self.embed_dim, self.n_heads)
        self.pre_mlp_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            SwishGLU(self.embed_dim, int(mlp_ratio * embed_dim)),
            nn.Linear(int(mlp_ratio * embed_dim), embed_dim),
        )

    def forward(self, seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq = seq + self.attn(self.pre_attn_norm(seq), mask)
        seq = seq + self.mlp(self.pre_mlp_norm(seq))
        return seq
