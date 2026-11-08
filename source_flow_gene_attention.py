"""Row-separable gene-attention backbone for conditional source flow models.

Each spot is processed independently.  Within a spot, genes are represented as
tokens and self-attention is applied over the gene axis.  In particular, the
output for row ``i`` cannot depend on any other batch row, which is the contract
required when a source vector field is interpreted pointwise as ``u_t(x, c)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _modulate_tokens(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class TimestepEmbedder(nn.Module):
    """Sinusoidal scalar-time embedding followed by a small MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        if hidden_size < 1 or frequency_embedding_size < 2:
            raise ValueError("time embedding dimensions must be positive.")
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = int(frequency_embedding_size)

    @staticmethod
    def timestep_embedding(
        timesteps: torch.Tensor, dimension: int, max_period: int = 10_000
    ) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape [batch].")
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(
                start=0,
                end=half,
                dtype=torch.float32,
                device=timesteps.device,
            )
            / half
        )
        arguments = timesteps[:, None].float() * frequencies[None]
        embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
        if dimension % 2:
            embedding = torch.cat(
                (embedding, torch.zeros_like(embedding[:, :1])), dim=-1
            )
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(
            self.timestep_embedding(timesteps, self.frequency_embedding_size)
        )


class GeneAttention(nn.Module):
    """Multi-head SDPA over genes with conceptual logits ``[B,H,G,G]``."""

    def __init__(self, hidden_size: int, *, num_heads: int) -> None:
        super().__init__()
        if hidden_size < 1 or num_heads < 1 or hidden_size % num_heads:
            raise ValueError("hidden_size must be positive and divisible by num_heads.")
        self.num_heads = int(num_heads)
        self.head_size = hidden_size // num_heads
        self.scale = self.head_size**-0.5
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("gene-attention tokens must have shape [B,G,H].")
        batch, genes, hidden = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, genes, 3, self.num_heads, self.head_size
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # This is mathematically the same scaled-softmax attention as the
        # explicit implementation, but lets PyTorch dispatch to FlashAttention
        # or a memory-efficient fused CUDA kernel when the inputs support it.
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        attended = attended.transpose(1, 2).reshape(batch, genes, hidden)
        return self.proj(attended)


class GeneDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")
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

    def forward(self, tokens: torch.Tensor, time_condition: torch.Tensor) -> torch.Tensor:
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(time_condition).chunk(6, dim=-1)
        tokens = tokens + gate_attention[:, None, :] * self.attn(
            _modulate_tokens(self.norm1(tokens), shift_attention, scale_attention)
        )
        tokens = tokens + gate_mlp[:, None, :] * self.mlp(
            _modulate_tokens(self.norm2(tokens), shift_mlp, scale_mlp)
        )
        return tokens


class GeneFinalLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, tokens: torch.Tensor, time_condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(time_condition).chunk(2, dim=-1)
        tokens = _modulate_tokens(self.norm(tokens), shift, scale)
        return self.linear(tokens).squeeze(-1)


class GeneAttentionSourceFlowBackbone(nn.Module):
    """Conditional DiT-style vector field with attention only across genes."""

    row_separable = True
    attention_axis = "genes"

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        condition_size: int | None = None,
    ) -> None:
        super().__init__()
        if min(input_size, hidden_size, depth, num_heads) < 1:
            raise ValueError("model dimensions must be positive.")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        resolved_condition_size = (
            input_size if condition_size is None else int(condition_size)
        )
        if resolved_condition_size < 1 or resolved_condition_size % input_size:
            raise ValueError(
                "condition_size must be a positive multiple of input_size so "
                "condition channels can be aligned gene by gene."
            )
        self.input_size = int(input_size)
        self.condition_size = resolved_condition_size
        self.condition_channels = resolved_condition_size // input_size
        self.hidden_size = int(hidden_size)
        self.value_embedding = nn.Linear(
            1 + self.condition_channels, hidden_size, bias=True
        )
        self.gene_embedding = nn.Embedding(input_size, hidden_size)
        self.time_emb = TimestepEmbedder(hidden_size=hidden_size)
        # Keep ``blks`` as the module name used by the earlier experimental
        # gene-attention implementation so those state_dict keys remain loadable.
        self.blks = nn.ModuleList(
            GeneDiTBlock(
                hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
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
        **_: object,
    ) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(f"x must have shape [batch,{self.input_size}].")
        if y is None or tuple(y.shape) != (x.shape[0], self.condition_size):
            raise ValueError(
                f"y must have shape [batch,{self.condition_size}]."
            )
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("t must have shape [batch].")

        # No operation below reduces over or attends across the batch axis.
        condition_channels = y.float().reshape(
            x.shape[0], self.condition_channels, self.input_size
        ).transpose(1, 2)
        token_values = torch.cat((x.float().unsqueeze(-1), condition_channels), dim=-1)
        tokens = self.value_embedding(token_values)
        gene_ids = torch.arange(self.input_size, device=x.device)
        tokens = tokens + self.gene_embedding(gene_ids)[None, :, :]
        time_condition = self.time_emb(t)
        for block in self.blks:
            tokens = block(tokens, time_condition)
        return self.out_layer(tokens, time_condition)


def build_gene_attention_source_flow_backbone(
    *,
    input_size: int,
    hidden_size: int,
    depth: int = 6,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
    condition_size: int | None = None,
) -> GeneAttentionSourceFlowBackbone:
    return GeneAttentionSourceFlowBackbone(
        input_size=input_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        condition_size=condition_size,
    )


__all__ = [
    "GeneAttention",
    "GeneAttentionSourceFlowBackbone",
    "build_gene_attention_source_flow_backbone",
]
