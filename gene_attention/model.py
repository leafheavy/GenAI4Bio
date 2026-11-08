"""stDiff denoisers with selectable gene and legacy batch attention."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.nn import functional as F

if __package__ and "." in __package__:
    from .. import model_original as _SOURCE
else:
    import model_original as _SOURCE

GENE_ATTENTION = "gene_attention"
BATCH_ATTENTION = "batch_attention"

# Reuse every non-stDiff component required by the existing training and
# inference pipelines.
Attention2 = _SOURCE.Attention2
DiTBlock = _SOURCE.DiTBlock
FinalLayer = _SOURCE.FinalLayer
IMDTModel = _SOURCE.IMDTModel
NoiseScheduler = _SOURCE.NoiseScheduler
SpatialGraphSmoothing = _SOURCE.SpatialGraphSmoothing
TimestepEmbedder = _SOURCE.TimestepEmbedder
betas_for_alpha_bar = _SOURCE.betas_for_alpha_bar
build_noise_scheduler = _SOURCE.build_noise_scheduler
get_schedule_jump = _SOURCE.get_schedule_jump
modulate = _SOURCE.modulate


def _modulate_tokens(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class GeneAttention(nn.Module):
    """Multi-head attention with logits shaped [batch, heads, genes, genes]."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, genes, dim = x.shape
        qkv = self.qkv(x).reshape(batch, genes, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if hasattr(F, "scaled_dot_product_attention"):
            attended = F.scaled_dot_product_attention(query, key, value)
        else:
            weights = ((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            attended = weights @ value
        x = attended.transpose(1, 2).reshape(batch, genes, dim)
        return self.proj(x)


class GeneDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = GeneAttention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_attn[:, None, :] * self.attn(
            _modulate_tokens(self.norm1(x), shift_attn, scale_attn)
        )
        x = x + gate_mlp[:, None, :] * self.mlp(
            _modulate_tokens(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class GeneFinalLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = _modulate_tokens(self.norm(x), shift, scale)
        return self.linear(x).squeeze(-1)


class StDiffModel(nn.Module):
    """Same stDiff call interface, with each gene represented as one token."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        depth: int,
        dit_type: str = "dit",
        num_heads: int = 16,
        classes: int = 6,
        mlp_ratio: float = 4.0,
        condition_size: int | None = None,
    ) -> None:
        super().__init__()
        if dit_type != "dit":
            raise ValueError("Only dit_type='dit' is supported.")
        if condition_size not in (None, input_size):
            raise ValueError("condition_size must equal input_size.")
        del classes  # Kept only for drop-in API compatibility.

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.value_embedding = nn.Linear(2, hidden_size, bias=True)
        self.gene_embedding = nn.Embedding(input_size, hidden_size)
        self.time_emb = TimestepEmbedder(hidden_size=hidden_size)
        self.blks = nn.ModuleList(
            [
                GeneDiTBlock(
                    hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.out_layer = GeneFinalLayer(hidden_size)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        nn.init.normal_(self.gene_embedding.weight, std=0.02)
        nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)
        for block in self.blks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.out_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.out_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.out_layer.linear.weight)
        nn.init.zeros_(self.out_layer.linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(f"x must have shape [batch, {self.input_size}].")
        if y is None or y.shape != x.shape:
            raise ValueError("y must have the same shape as x.")

        # Each [x_t(gene), condition(gene)] pair becomes one gene token.
        tokens = self.value_embedding(torch.stack((x.float(), y.float()), dim=-1))
        gene_ids = torch.arange(self.input_size, device=x.device)
        tokens = tokens + self.gene_embedding(gene_ids)[None, :, :]
        condition = self.time_emb(t)
        for block in self.blks:
            tokens = block(tokens, condition)
        return self.out_layer(tokens, condition)


def resolve_attention_architecture(method: str, architecture: str | None = None) -> str:
    method = method.strip().lower()
    if method not in {"stdiff", "imdt"}:
        raise ValueError("method must be 'stdiff' or 'imdt'.")
    if architecture is None:
        return GENE_ATTENTION if method == "stdiff" else BATCH_ATTENTION
    if architecture not in {GENE_ATTENTION, BATCH_ATTENTION}:
        raise ValueError(f"Unknown attention architecture: {architecture!r}.")
    if method == "imdt" and architecture != BATCH_ATTENTION:
        raise ValueError("Gene attention is supported for stDiff only.")
    return architecture


def checkpoint_attention_architecture(
    method: str,
    model_config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> str:
    state_dict = checkpoint["model_state_dict"]
    inferred = (
        GENE_ATTENTION if "gene_embedding.weight" in state_dict else BATCH_ATTENTION
    )
    resolve_attention_architecture(method, inferred)
    for config in (model_config, checkpoint.get("model_config", {})):
        architecture = config.get("architecture")
        if architecture is not None:
            resolve_attention_architecture(method, architecture)
            if architecture != inferred:
                raise ValueError(
                    f"Recorded architecture {architecture!r} disagrees with "
                    f"checkpoint weights ({inferred!r})."
                )
    return inferred


def build_model(
    method: str,
    *,
    input_size: int,
    hidden_size: int,
    depth: int = 6,
    num_heads: int = 16,
    classes: int | None = None,
    mlp_ratio: float = 4.0,
    condition_size: int | None = None,
    architecture: str | None = None,
) -> nn.Module:
    if resolve_attention_architecture(method, architecture) == GENE_ATTENTION:
        return StDiffModel(
            input_size=input_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            classes=6 if classes is None else classes,
            mlp_ratio=mlp_ratio,
            condition_size=condition_size,
        )
    return _SOURCE.build_model(
        method,
        input_size=input_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        classes=classes,
        mlp_ratio=mlp_ratio,
        condition_size=condition_size,
    )


__all__ = [
    "BATCH_ATTENTION",
    "GENE_ATTENTION",
    "checkpoint_attention_architecture",
    "resolve_attention_architecture",
    "Attention2",
    "DiTBlock",
    "FinalLayer",
    "GeneAttention",
    "GeneDiTBlock",
    "GeneFinalLayer",
    "IMDTModel",
    "NoiseScheduler",
    "SpatialGraphSmoothing",
    "StDiffModel",
    "TimestepEmbedder",
    "betas_for_alpha_bar",
    "build_model",
    "build_noise_scheduler",
    "get_schedule_jump",
    "modulate",
]
