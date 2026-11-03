"""A small Llama-style decoder-only transformer.

Modern-LLM building blocks, all cheap on CPU, MPS and CUDA alike:
  - pre-norm RMSNorm, no biases anywhere
  - rotary position embeddings (RoPE)
  - SwiGLU feed-forward
  - grouped-query attention (n_kv_head <= n_head) and optional QK-norm
  - tied input/output embeddings
  - fused scaled_dot_product_attention (causal)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 4
    n_kv_head: int = 4
    d_model: int = 256
    ctx_len: int = 256
    mlp_hidden: int = 0  # 0 -> ~8/3 * d_model rounded to a multiple of 64 (SwiGLU)
    rope_theta: float = 10000.0
    qk_norm: bool = False
    tie_embeddings: bool = True
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        assert (self.d_model // self.n_head) % 2 == 0, "head dim must be even for RoPE"
        if self.mlp_hidden == 0:
            self.mlp_hidden = int(round(self.d_model * 8 / 3 / 64)) * 64


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


def rope_tables(seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()  # (T, head_dim / 2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Half-split rotation (GPT-NeoX / Llama convention). x: (B, H, T, D)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head, self.n_kv_head = cfg.n_head, cfg.n_kv_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.qk_norm else None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class LLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_out = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = rope_tables(cfg.ctx_len, cfg.d_model // cfg.n_head, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        # GPT-2 style scaling of residual-branch output projections.
        resid_std = std / (2 * self.cfg.n_layer) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=resid_std)
            nn.init.normal_(block.mlp.w_down.weight, mean=0.0, std=resid_std)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        T = idx.size(1)
        assert T <= self.cfg.ctx_len, f"sequence length {T} > ctx_len {self.cfg.ctx_len}"
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm_out(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss

    def num_params(self) -> int:
        # parameters() de-duplicates tied weights, so this counts the embedding once.
        return sum(p.numel() for p in self.parameters())

    def num_non_embedding_params(self) -> int:
        return self.num_params() - self.tok_emb.weight.numel()

    def flops_per_token(self, seq_len: int) -> float:
        """Approximate training FLOPs per token: 6N for the matmuls plus the attention term."""
        n_layer, d = self.cfg.n_layer, self.cfg.d_model
        return 6 * self.num_params() + 12 * n_layer * d * seq_len
