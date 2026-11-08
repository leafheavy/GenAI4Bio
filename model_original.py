"""stDiff and IMD-T model definitions.

The DiT blocks, initialisation, cosine schedule, reverse equations, IMD-T class
conditioning, and SG operator are migrated from the two prior-work directories
with their numerical behaviour kept intact.  File paths and training loops do
not belong in this module.
"""

from __future__ import annotations

import math
from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from timm.models.vision_transformer import Mlp
from torch.nn import functional as F


class Attention2(nn.Module):
    """The prior-work attention implementation (attention across batch rows)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim should be divisible by num_heads")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cells, features = x.shape
        qkv = self.qkv(x).reshape(
            cells, 3, self.num_heads, features // self.num_heads
        )
        qkv = einops.rearrange(qkv, "c n h fph -> n h c fph")
        query, key, value = qkv.unbind(0)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ value).transpose(0, 1)
        x = einops.rearrange(x, "c h fph -> c (h fph)")
        return self.proj_drop(self.proj(x))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        timesteps: torch.Tensor, dim: int, max_period: int = 10_000
    ) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        frequency = self.timestep_embedding(
            timesteps, self.frequency_embedding_size
        )
        return self.mlp(frequency)


class DiTBlock(nn.Module):
    def __init__(
        self,
        feature_dim: int = 2000,
        mlp_ratio: float = 4.0,
        num_heads: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(
            feature_dim, elementwise_affine=False, eps=1e-6
        )
        self.attn = Attention2(
            feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs
        )
        self.norm2 = nn.LayerNorm(
            feature_dim, elementwise_affine=False, eps=1e-6
        )
        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=feature_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(feature_dim, 6 * feature_dim, bias=True)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(condition).chunk(6, dim=1)
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class _BaseDiT(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        depth: int,
        dit_type: str,
        num_heads: int,
        classes: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if dit_type != "dit":
            raise ValueError("The migrated prior work only defines dit_type='dit'.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.classes = classes
        self.mlp_ratio = mlp_ratio
        self.dit_type = dit_type
        self.in_layer = nn.Sequential(nn.Linear(input_size, hidden_size))
        self.time_emb = TimestepEmbedder(hidden_size=hidden_size)
        self.blks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads,
                )
                for _ in range(depth)
            ]
        )
        self.out_layer = FinalLayer(hidden_size, input_size)

    def _basic_initialise(self) -> None:
        def initialise(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(initialise)
        nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)
        for block in self.blks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.linear.weight, 0)
        nn.init.constant_(self.out_layer.linear.bias, 0)

    def _denoise(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.in_layer(x.float())
        for block in self.blks:
            x = block(x, condition)
        return self.out_layer(x, condition)


class StDiffModel(_BaseDiT):
    """Original stDiff DiT with a continuous masked-expression condition."""

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
        super().__init__(
            input_size,
            hidden_size,
            depth,
            dit_type,
            num_heads,
            classes,
            mlp_ratio,
        )
        self.condition_size = input_size if condition_size is None else condition_size
        if self.condition_size < 1:
            raise ValueError("condition_size must be positive.")
        self.cond_layer = nn.Sequential(nn.Linear(self.condition_size, hidden_size))
        # Retained for state compatibility with the upstream stDiff model.
        self.condi_emb = nn.Embedding(classes, hidden_size)
        self._basic_initialise()
        nn.init.normal_(self.condi_emb.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        if y is None:
            raise ValueError("stDiff requires the masked expression condition y.")
        condition = self.time_emb(t) + self.cond_layer(y)
        return self._denoise(x, condition)


class IMDTModel(_BaseDiT):
    """IMD-T DiT with the original zero-initialised class embedding."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        depth: int,
        dit_type: str = "dit",
        num_heads: int = 16,
        classes: int = 6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__(
            input_size,
            hidden_size,
            depth,
            dit_type,
            num_heads,
            classes,
            mlp_ratio,
        )
        self.condi_emb = nn.Embedding(classes, hidden_size)
        self._basic_initialise()
        nn.init.zeros_(self.condi_emb.weight)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        condition = self.time_emb(t)
        if y is not None:
            condition = condition + self.condi_emb(y.long())
        return self._denoise(x, condition)


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
) -> nn.Module:
    method_key = method.strip().lower()
    common = dict(
        input_size=input_size,
        hidden_size=hidden_size,
        depth=depth,
        dit_type="dit",
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )
    if method_key == "stdiff":
        return StDiffModel(
            classes=6 if classes is None else classes,
            condition_size=condition_size,
            **common,
        )
    if method_key == "imdt":
        if condition_size is not None:
            raise ValueError("condition_size is only supported by stDiff.")
        if classes is None or classes < 1:
            raise ValueError("IMD-T requires the number of scRNA Leiden classes.")
        return IMDTModel(classes=classes, **common)
    raise ValueError("method must be 'stdiff' or 'imdt'.")


def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    alpha_bar: Any,
    max_beta: float = 0.999,
) -> np.ndarray:
    betas = []
    for index in range(num_diffusion_timesteps):
        t1 = index / num_diffusion_timesteps
        t2 = (index + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.asarray(betas)


def _extract_into_tensor(
    array: np.ndarray, timesteps: torch.Tensor, broadcast_shape: torch.Size | tuple[int, ...]
) -> torch.Tensor:
    result = torch.from_numpy(array).to(device=timesteps.device)[timesteps].float()
    while len(result.shape) < len(broadcast_shape):
        result = result[..., None]
    return result.expand(broadcast_shape)


class NoiseScheduler:
    """Shared diffusion equations with method-specific reverse stochasticity."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        *,
        stochastic_reverse: bool = True,
        clip_denoised: bool = True,
    ) -> None:
        self.num_timesteps = num_timesteps
        self.stochastic_reverse = stochastic_reverse
        self.clip_denoised = clip_denoised
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_timesteps,
                dtype=torch.float32,
            ) ** 2
        elif beta_schedule == "cosine":
            self.betas = torch.from_numpy(
                betas_for_alpha_bar(
                    num_timesteps,
                    lambda t: math.cos(
                        (t + 0.008) / 1.008 * math.pi / 2
                    )
                    ** 2,
                ).astype(np.float32)
            )
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self.sqrt_alphas_cumprod = self.alphas_cumprod**0.5
        self.sqrt_one_minus_alphas_cumprod = (
            1 - self.alphas_cumprod
        ) ** 0.5
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1
        )
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.gt_noises: torch.Tensor | None = None

    @staticmethod
    def _timestep_value(timestep: int | torch.Tensor) -> int:
        if isinstance(timestep, torch.Tensor):
            return int(timestep.reshape(-1)[0].item())
        return int(timestep)

    def reconstruct_x0(
        self, x_t: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        s1 = self.sqrt_inv_alphas_cumprod[timestep]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[timestep]
        s1 = s1.reshape(-1, 1).to(x_t.device)
        s2 = s2.reshape(-1, 1).to(x_t.device)
        estimate = s1 * x_t - s2 * noise
        return torch.clamp(estimate, min=-1, max=1) if self.clip_denoised else estimate

    def q_posterior(
        self, x_0: torch.Tensor, x_t: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        s1 = self.posterior_mean_coef1[timestep].reshape(-1, 1).to(x_t.device)
        s2 = self.posterior_mean_coef2[timestep].reshape(-1, 1).to(x_t.device)
        return s1 * x_0 + s2 * x_t

    def get_variance(self, timestep: int | torch.Tensor) -> torch.Tensor | float:
        value = self._timestep_value(timestep)
        if value == 0:
            return 0.0
        variance = (
            self.betas[value]
            * (1.0 - self.alphas_cumprod_prev[value])
            / (1.0 - self.alphas_cumprod[value])
        ).clip(1e-20)
        if isinstance(timestep, torch.Tensor):
            variance = variance.to(timestep.device)
        return variance

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        model_pred_type: str = "noise",
        sample_pred_xstart: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if model_pred_type == "noise":
            pred_original_sample = self.reconstruct_x0(
                sample, timestep, model_output
            )
        elif model_pred_type == "x_start":
            pred_original_sample = model_output
        else:
            raise NotImplementedError(model_pred_type)

        pred_prev_sample = self.q_posterior(
            pred_original_sample, sample, timestep
        )
        add_noise = (
            self.stochastic_reverse
            if sample_pred_xstart is None
            else sample_pred_xstart
        )
        if add_noise and self._timestep_value(timestep) > 0:
            variance = self.get_variance(timestep)
            pred_prev_sample = pred_prev_sample + variance**0.5 * torch.randn_like(
                model_output
            )
        return pred_prev_sample, pred_original_sample

    def add_noise(
        self,
        x_start: torch.Tensor,
        x_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        s1 = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1).to(x_start.device)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1).to(
            x_start.device
        )
        return s1 * x_start + s2 * x_noise

    def q_sample(
        self, x_start: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x_start)
        return self.add_noise(x_start, noise, timesteps), noise

    def store_gt_noises(
        self, _x_start: torch.Tensor, noises: torch.Tensor
    ) -> None:
        self.gt_noises = noises

    def get_gt_noised(self, gt: torch.Tensor, timestep: int) -> torch.Tensor:
        device = gt.device
        t_tensor = torch.tensor([timestep], device=device)
        noise = (
            self.gt_noises.to(device)
            if self.gt_noises is not None
            else torch.randn_like(gt)
        )
        # Kept identical to the experimental IMD-T RePaint scheduler.
        sqrt_alphas_cumprod_t = _extract_into_tensor(
            self.alphas_cumprod.cpu().numpy(), t_tensor.cpu(), gt.shape
        ).to(device)
        sqrt_one_minus_t = torch.sqrt(1 - sqrt_alphas_cumprod_t**2)
        return sqrt_alphas_cumprod_t * gt + sqrt_one_minus_t * noise

    def reset_gt_noises(self) -> None:
        self.gt_noises = None

    def undo(
        self,
        image_before_step: torch.Tensor,
        img_after_model: torch.Tensor,
        est_x_0: torch.Tensor | None,
        timestep: int | torch.Tensor,
        debug: bool = False,
    ) -> torch.Tensor:
        del image_before_step, est_x_0, debug
        return self._undo(img_after_model, timestep)

    def _undo(
        self, img_out: torch.Tensor, timestep: int | torch.Tensor
    ) -> torch.Tensor:
        value = self._timestep_value(timestep)
        t_plus_1 = value + 1 if value + 1 < self.num_timesteps else value
        tensor = torch.tensor(
            [t_plus_1], device=img_out.device, dtype=torch.long
        )
        coef1 = self.posterior_mean_coef1[tensor].reshape(-1, 1).to(img_out.device)
        coef2 = self.posterior_mean_coef2[tensor].reshape(-1, 1).to(img_out.device)
        estimate = coef1 * img_out + coef2 * img_out
        variance = self.get_variance(tensor)
        return estimate + variance**0.5 * torch.randn_like(img_out)

    def __len__(self) -> int:
        return self.num_timesteps


def build_noise_scheduler(
    method: str,
    num_timesteps: int,
    *,
    clip_denoised: bool = True,
) -> NoiseScheduler:
    method_key = method.strip().lower()
    if method_key not in {"stdiff", "imdt"}:
        raise ValueError("method must be 'stdiff' or 'imdt'.")
    return NoiseScheduler(
        num_timesteps=num_timesteps,
        beta_schedule="cosine",
        stochastic_reverse=(method_key == "stdiff"),
        clip_denoised=clip_denoised,
    )


def get_schedule_jump(
    t_T: int,
    n_sample: int,
    jump_length: int,
    jump_n_sample: int,
    jump2_length: int = 1,
    jump2_n_sample: int = 1,
    jump3_length: int = 1,
    jump3_n_sample: int = 1,
    start_resampling: int = 100_000_000,
) -> list[int]:
    """Experimental IMD-T RePaint time-jump schedule, migrated verbatim."""

    jumps = {
        value: jump_n_sample - 1
        for value in range(0, t_T - jump_length, jump_length)
    }
    jumps2 = {
        value: jump2_n_sample - 1
        for value in range(0, t_T - jump2_length, jump2_length)
    }
    jumps3 = {
        value: jump3_n_sample - 1
        for value in range(0, t_T - jump3_length, jump3_length)
    }
    timestep = t_T
    times: list[int] = []
    while timestep >= 1:
        timestep -= 1
        times.append(timestep)
        if timestep + 1 < t_T - 1 and timestep <= start_resampling:
            for _ in range(n_sample - 1):
                timestep += 1
                times.append(timestep)
                if timestep >= 0:
                    timestep -= 1
                    times.append(timestep)
        if (
            jumps3.get(timestep, 0) > 0
            and timestep <= start_resampling - jump3_length
        ):
            jumps3[timestep] -= 1
            for _ in range(jump3_length):
                timestep += 1
                times.append(timestep)
        if (
            jumps2.get(timestep, 0) > 0
            and timestep <= start_resampling - jump2_length
        ):
            jumps2[timestep] -= 1
            for _ in range(jump2_length):
                timestep += 1
                times.append(timestep)
            jumps3 = {
                value: jump3_n_sample - 1
                for value in range(0, t_T - jump3_length, jump3_length)
            }
        if (
            jumps.get(timestep, 0) > 0
            and timestep <= start_resampling - jump_length
        ):
            jumps[timestep] -= 1
            for _ in range(jump_length):
                timestep += 1
                times.append(timestep)
            jumps2 = {
                value: jump2_n_sample - 1
                for value in range(0, t_T - jump2_length, jump2_length)
            }
            jumps3 = {
                value: jump3_n_sample - 1
                for value in range(0, t_T - jump3_length, jump3_length)
            }
    times.append(-1)
    if not times[0] > times[1] or times[-1] != -1:
        raise AssertionError("Invalid RePaint schedule endpoints.")
    for previous, current in zip(times[:-1], times[1:]):
        if abs(previous - current) != 1:
            raise AssertionError((previous, current))
    return times


class SpatialGraphSmoothing:
    """Directed KNN smoothing used by the default IMD-T sampler."""

    def __init__(
        self,
        coordinates: np.ndarray,
        K: int = 6,
        device: str | torch.device = "cuda",
    ) -> None:
        self.K = K
        self.device = torch.device(device)
        self.N = coordinates.shape[0]
        if self.N < K + 1:
            raise ValueError(f"K={K} smoothing requires at least {K + 1} spots.")
        self.A = self.build_knn_graph(coordinates)

    def build_knn_graph(self, coordinates: np.ndarray) -> torch.Tensor:
        knn = NearestNeighbors(n_neighbors=self.K + 1, metric="euclidean")
        knn.fit(coordinates)
        _, indices = knn.kneighbors(coordinates)
        indices = indices[:, 1:]
        rows: list[int] = []
        cols: list[int] = []
        for index in range(self.N):
            for neighbour in indices[index]:
                rows.append(index)
                cols.append(int(neighbour))
        row_tensor = torch.tensor(rows, dtype=torch.long)
        col_tensor = torch.tensor(cols, dtype=torch.long)
        values = torch.ones(len(rows))
        graph = torch.sparse_coo_tensor(
            torch.stack([row_tensor, col_tensor]),
            values,
            (self.N, self.N),
        )
        return graph.coalesce().to(self.device)

    def SG(self, expression: torch.Tensor) -> torch.Tensor:
        device = expression.device
        identity_indices = torch.arange(self.N, device=device)
        identity = torch.sparse_coo_tensor(
            torch.stack([identity_indices, identity_indices]),
            torch.ones(self.N, device=device),
            (self.N, self.N),
        )
        graph_with_self = self.A.to(device) + identity
        smoothed = torch.sparse.mm(graph_with_self, expression)
        return smoothed / (self.K + 1)


__all__ = [
    "Attention2",
    "DiTBlock",
    "FinalLayer",
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
