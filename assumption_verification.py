"""Verify SC-to-ST condition and conditional-score shifts for stDiff.

This is intentionally a single-file experiment.  It trains two controlled
stDiff models per gene fold:

1. an SC model trained on unaugmented single-cell profiles;
2. an ST-oracle model trained on unaugmented, held-in spatial spots.

The models share their architecture, initial weights, sample count, batch
order, diffusion timesteps, and Gaussian noises.  They are compared on the
same held-out ST spots.  Missing genes are available as training targets, but
never as condition inputs.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm import tqdm

from Dataloader import (
    DEFAULT_DATA_ROOT,
    build_gene_folds,
    load_paired_h5ad,
    resolve_dataset_paths,
    scale,
)
from model_original import build_model, build_noise_scheduler
from utils import ensure_output_dir, load_json, resolve_device, save_json, set_seed


STDIFF_DEFAULT_LR = 0.00016046744893538737


@dataclass(frozen=True)
class ExperimentSettings:
    """Compact configuration for the controlled two-model experiment."""

    n_splits: int = 5
    fold: int = 0  # 0 runs every gene fold.
    spot_test_fraction: float = 0.2
    split_seed: int = 0
    model_seed: int = 1202
    batch_size: int = 512
    epochs: int = 900
    learning_rate: float = STDIFF_DEFAULT_LR
    diffusion_steps: int = 1500
    hidden_size: int = 1024
    depth: int = 6
    num_heads: int = 16
    mlp_ratio: float = 4.0
    max_train_samples: int = 0
    max_mmd_samples: int = 1000
    max_plot_points: int = 2000
    max_probe_spots: int = 512
    arrow_spots: int = 40
    time_fractions: tuple[float, ...] = (0.1, 0.5, 0.9)
    deterministic: bool = False
    show_progress: bool = True


def _dense_float32(matrix: object) -> np.ndarray:
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def preprocess_without_augmentation(
    adata_sc: sc.AnnData,
    adata_st: sc.AnnData,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the stDiff numerical transform without SC-only augmentation."""

    processed: list[np.ndarray] = []
    for adata in (adata_sc.copy(), adata_st.copy()):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = scale(adata)
        processed.append(_dense_float32(adata.X))
    return processed[0], processed[1]


def split_and_match_samples(
    n_sc: int,
    n_st: int,
    settings: ExperimentSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold out ST spots and match SC/ST training sample counts."""

    if n_st < 3:
        raise ValueError("At least three ST spots are required for train/test splitting.")
    if not 0.0 < settings.spot_test_fraction < 1.0:
        raise ValueError("spot_test_fraction must be between 0 and 1.")

    rng = np.random.default_rng(settings.split_seed)
    st_permutation = rng.permutation(n_st)
    n_test = int(round(n_st * settings.spot_test_fraction))
    n_test = min(max(n_test, 1), n_st - 2)
    st_test_indices = np.sort(st_permutation[:n_test])
    st_train_pool = st_permutation[n_test:]

    matched_count = min(n_sc, len(st_train_pool))
    if settings.max_train_samples > 0:
        matched_count = min(matched_count, settings.max_train_samples)
    if matched_count < 2:
        raise ValueError("At least two matched SC/ST training samples are required.")

    sc_train_indices = rng.choice(n_sc, matched_count, replace=False)
    st_train_indices = rng.choice(st_train_pool, matched_count, replace=False)
    return sc_train_indices, st_train_indices, st_test_indices


def build_fold_arrays(
    sc_values: np.ndarray,
    st_values: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return diffusion targets and masked conditions in stDiff's [-1, 1] range."""

    sc_expression = sc_values * 2.0 - 1.0
    st_expression = st_values * 2.0 - 1.0
    sc_condition = (sc_values * mask) * 2.0 - 1.0
    st_condition = (st_values * mask) * 2.0 - 1.0
    return sc_expression, sc_condition, st_expression, st_condition


def rbf_mmd_squared(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> tuple[float, float]:
    """Unbiased RBF-MMD^2 with a pooled median-bandwidth heuristic."""

    rng = np.random.default_rng(seed)
    if max_samples > 0:
        if len(x) > max_samples:
            x = x[rng.choice(len(x), max_samples, replace=False)]
        if len(y) > max_samples:
            y = y[rng.choice(len(y), max_samples, replace=False)]
    if len(x) < 2 or len(y) < 2:
        raise ValueError("MMD requires at least two samples from each distribution.")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        distances = (
            np.sum(a * a, axis=1, keepdims=True)
            + np.sum(b * b, axis=1)[None, :]
            - 2.0 * (a @ b.T)
        )
        return np.maximum(distances, 0.0)

    pooled = np.concatenate([x, y], axis=0)
    pooled_distances = squared_distances(pooled, pooled)
    upper = pooled_distances[np.triu_indices(len(pooled), k=1)]
    positive = upper[upper > 0]
    bandwidth_squared = float(np.median(positive)) if len(positive) else 1.0
    bandwidth_squared = max(bandwidth_squared, np.finfo(np.float64).eps)

    k_xx = np.exp(-squared_distances(x, x) / (2.0 * bandwidth_squared))
    k_yy = np.exp(-squared_distances(y, y) / (2.0 * bandwidth_squared))
    k_xy = np.exp(-squared_distances(x, y) / (2.0 * bandwidth_squared))
    m, n = len(x), len(y)
    mmd_squared = (
        (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
        + (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
        - 2.0 * k_xy.mean()
    )
    # The unbiased estimate can be slightly negative because of finite samples.
    return float(max(mmd_squared, 0.0)), bandwidth_squared


def plot_condition_shift(
    sc_condition: np.ndarray,
    st_condition: np.ndarray,
    observed_indices: np.ndarray,
    *,
    mmd_squared: float,
    output_path: Path,
    max_points: int,
    seed: int,
) -> float:
    """Plot both condition distributions in one jointly fitted PCA space."""

    sc_observed = sc_condition[:, observed_indices]
    st_observed = st_condition[:, observed_indices]
    pooled = np.concatenate([sc_observed, st_observed], axis=0)
    pca = PCA(n_components=2, random_state=seed)
    coordinates = pca.fit_transform(pooled)
    sc_coordinates = coordinates[: len(sc_observed)]
    st_coordinates = coordinates[len(sc_observed) :]

    rng = np.random.default_rng(seed)

    def subset(values: np.ndarray) -> np.ndarray:
        if max_points > 0 and len(values) > max_points:
            return values[rng.choice(len(values), max_points, replace=False)]
        return values

    sc_plot = subset(sc_coordinates)
    st_plot = subset(st_coordinates)
    explained = float(pca.explained_variance_ratio_[:2].sum())

    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    axis.scatter(
        sc_plot[:, 0], sc_plot[:, 1], s=13, alpha=0.42,
        color="#D50000", label="SC training conditions",
    )
    axis.scatter(
        st_plot[:, 0], st_plot[:, 1], s=13, alpha=0.52,
        color="#0072B2", label="ST-oracle training conditions",
    )
    axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axis.set_title(f"Conditional-space shift\nRBF MMD² = {mmd_squared:.4g}")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return explained


def train_controlled_pair(
    sc_expression: np.ndarray,
    sc_condition: np.ndarray,
    st_expression: np.ndarray,
    st_condition: np.ndarray,
    mask: np.ndarray,
    *,
    settings: ExperimentSettings,
    device: torch.device,
    fold_number: int,
) -> tuple[torch.nn.Module, torch.nn.Module, pd.DataFrame, dict[str, object]]:
    """Train SC and ST models in lockstep with paired optimization randomness."""

    if len(sc_expression) != len(st_expression):
        raise ValueError("SC and ST training arrays must have matched sample counts.")

    set_seed(settings.model_seed, deterministic=settings.deterministic)
    model_config = {
        "input_size": int(sc_expression.shape[1]),
        "hidden_size": settings.hidden_size,
        "depth": settings.depth,
        "num_heads": settings.num_heads,
        "classes": 6,
        "mlp_ratio": settings.mlp_ratio,
    }
    sc_model = build_model("stdiff", **model_config)
    st_model = copy.deepcopy(sc_model)
    sc_model.to(device)
    st_model.to(device)

    sc_optimizer = torch.optim.AdamW(
        sc_model.parameters(), lr=settings.learning_rate, weight_decay=0
    )
    st_optimizer = torch.optim.AdamW(
        st_model.parameters(), lr=settings.learning_rate, weight_decay=0
    )
    scheduler = build_noise_scheduler("stdiff", settings.diffusion_steps)
    observed = torch.as_tensor(mask, dtype=torch.float32, device=device)
    missing = 1.0 - observed

    sc_x = torch.from_numpy(sc_expression)
    sc_y = torch.from_numpy(sc_condition)
    st_x = torch.from_numpy(st_expression)
    st_y = torch.from_numpy(st_condition)
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(settings.model_seed)
    history: list[dict[str, float | int]] = []

    epoch_iterator: Sequence[int] = range(settings.epochs)
    if settings.show_progress:
        epoch_iterator = tqdm(
            epoch_iterator, desc=f"fold {fold_number}: SC vs ST", ncols=100
        )

    sc_model.train()
    st_model.train()
    for epoch in epoch_iterator:
        permutation = torch.randperm(len(sc_x), generator=order_generator)
        sc_loss_sum = 0.0
        st_loss_sum = 0.0
        batch_count = 0

        for start in range(0, len(permutation), settings.batch_size):
            indices = permutation[start : start + settings.batch_size]
            sc_batch_x = sc_x[indices].to(device=device, dtype=torch.float32)
            sc_batch_y = sc_y[indices].to(device=device, dtype=torch.float32)
            st_batch_x = st_x[indices].to(device=device, dtype=torch.float32)
            st_batch_y = st_y[indices].to(device=device, dtype=torch.float32)

            # Reuse exactly the same t and epsilon for the paired SC/ST batches.
            timesteps = torch.randint(
                1, settings.diffusion_steps, (len(indices),), device="cpu"
            ).long()
            noise = torch.randn_like(sc_batch_x)

            def update_one(
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                expression: torch.Tensor,
                condition: torch.Tensor,
            ) -> float:
                noisy = scheduler.add_noise(expression, noise, timesteps)
                # This matches stDiff training: only missing genes are noised.
                model_input = noisy * missing + expression * observed
                optimizer.zero_grad(set_to_none=True)
                predicted_noise = model(
                    model_input, t=timesteps.to(device), y=condition
                )
                loss = F.mse_loss(
                    predicted_noise * missing,
                    noise * missing,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                return float(loss.item())

            sc_loss_sum += update_one(
                sc_model, sc_optimizer, sc_batch_x, sc_batch_y
            )
            st_loss_sum += update_one(
                st_model, st_optimizer, st_batch_x, st_batch_y
            )
            batch_count += 1

        sc_epoch_loss = sc_loss_sum / batch_count
        st_epoch_loss = st_loss_sum / batch_count
        history.append(
            {
                "epoch": epoch + 1,
                "sc_noise_loss": sc_epoch_loss,
                "st_oracle_noise_loss": st_epoch_loss,
            }
        )
        if settings.show_progress and isinstance(epoch_iterator, tqdm):
            epoch_iterator.set_postfix_str(
                f"SC {sc_epoch_loss:.5f} | ST {st_epoch_loss:.5f}"
            )

    return sc_model, st_model, pd.DataFrame(history), model_config


def load_controlled_pair(
    checkpoint_run_dir: Path,
    *,
    fold_number: int,
    genes: np.ndarray,
    missing_indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, object], dict[str, object]]:
    """Rebuild both models from one prior run without retraining them."""

    fold_dir = checkpoint_run_dir / f"fold_{fold_number:02d}"
    checkpoint_paths = {
        "SC": fold_dir / "sc_controlled_model.pt",
        "ST-oracle": fold_dir / "st_oracle_model.pt",
    }
    for path in checkpoint_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    def torch_load(path: Path) -> dict[str, object]:
        # weights_only=False is explicit because these trusted local files also
        # contain NumPy metadata in addition to tensor weights.
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # Compatibility with older PyTorch versions.
            return torch.load(path, map_location="cpu")

    sc_checkpoint = torch_load(checkpoint_paths["SC"])
    st_checkpoint = torch_load(checkpoint_paths["ST-oracle"])
    if sc_checkpoint.get("source") != "SC":
        raise ValueError(f"Unexpected SC checkpoint source in {checkpoint_paths['SC']}.")
    if st_checkpoint.get("source") != "ST-oracle":
        raise ValueError(
            f"Unexpected ST checkpoint source in {checkpoint_paths['ST-oracle']}."
        )

    for checkpoint, path in (
        (sc_checkpoint, checkpoint_paths["SC"]),
        (st_checkpoint, checkpoint_paths["ST-oracle"]),
    ):
        if int(checkpoint.get("fold", -1)) != fold_number:
            raise ValueError(f"Fold metadata mismatch in {path}.")
        if list(checkpoint.get("genes", [])) != genes.tolist():
            raise ValueError(f"Gene identity/order mismatch in {path}.")
        saved_missing = np.asarray(checkpoint.get("missing_indices"), dtype=np.int64)
        if not np.array_equal(saved_missing, missing_indices):
            raise ValueError(f"Missing-gene mask mismatch in {path}.")

    model_config = dict(sc_checkpoint["model_config"])
    if model_config != dict(st_checkpoint["model_config"]):
        raise ValueError("SC and ST-oracle checkpoint model configurations differ.")
    training_settings = dict(sc_checkpoint["settings"])
    if training_settings != dict(st_checkpoint["settings"]):
        raise ValueError("SC and ST-oracle checkpoint training settings differ.")

    sc_model = build_model("stdiff", **model_config)
    st_model = build_model("stdiff", **model_config)
    sc_model.load_state_dict(sc_checkpoint["model_state_dict"], strict=True)
    st_model.load_state_dict(st_checkpoint["model_state_dict"], strict=True)
    return (
        sc_model.to(device),
        st_model.to(device),
        model_config,
        training_settings,
    )


@torch.no_grad()
def evaluate_score_shift(
    sc_model: torch.nn.Module,
    st_model: torch.nn.Module,
    st_test_expression: np.ndarray,
    st_test_condition: np.ndarray,
    missing_indices: np.ndarray,
    mask: np.ndarray,
    *,
    settings: ExperimentSettings,
    device: torch.device,
    fold_number: int,
) -> tuple[pd.DataFrame, list[dict[str, np.ndarray | float | int]]]:
    """Compare conditional scores on identical held-out ST diffusion probes."""

    rng = np.random.default_rng(settings.split_seed + 10_000)
    probe_count = len(st_test_expression)
    if settings.max_probe_spots > 0:
        probe_count = min(probe_count, settings.max_probe_spots)
    probe_indices = rng.choice(len(st_test_expression), probe_count, replace=False)
    expression_np = st_test_expression[probe_indices]
    condition_np = st_test_condition[probe_indices]
    probe_noise_np = rng.standard_normal(expression_np.shape).astype(np.float32)

    expression = torch.from_numpy(expression_np).to(device)
    condition = torch.from_numpy(condition_np).to(device)
    probe_noise = torch.from_numpy(probe_noise_np).to(device)
    observed = torch.as_tensor(mask, dtype=torch.float32, device=device)
    missing = 1.0 - observed
    missing_index_tensor = torch.as_tensor(
        missing_indices, dtype=torch.long, device=device
    )
    scheduler = build_noise_scheduler("stdiff", settings.diffusion_steps)

    timesteps = sorted(
        {
            max(1, min(settings.diffusion_steps - 1,
                       int(round(fraction * (settings.diffusion_steps - 1)))))
            for fraction in settings.time_fractions
        }
    )
    sc_model.eval()
    st_model.eval()
    records: list[dict[str, float | int]] = []
    arrow_payload: list[dict[str, np.ndarray | float | int]] = []

    for timestep in timesteps:
        timestep_batch = torch.full(
            (probe_count,), timestep, dtype=torch.long, device="cpu"
        )
        noisy = scheduler.add_noise(expression, probe_noise, timestep_batch)
        model_input = noisy * missing + expression * observed
        model_t = timestep_batch.to(device)
        sc_noise = sc_model(model_input, t=model_t, y=condition)
        st_noise = st_model(model_input, t=model_t, y=condition)

        sigma_t = float(scheduler.sqrt_one_minus_alphas_cumprod[timestep])
        sc_score = -sc_noise[:, missing_index_tensor] / sigma_t
        st_score = -st_noise[:, missing_index_tensor] / sigma_t
        difference = torch.linalg.vector_norm(sc_score - st_score, dim=1)
        reference = torch.linalg.vector_norm(st_score, dim=1).clamp_min(1e-8)
        relative_gap = float((difference / reference).mean().item())

        target_noise = probe_noise[:, missing_index_tensor]
        sc_noise_mse = float(
            F.mse_loss(sc_noise[:, missing_index_tensor], target_noise).item()
        )
        st_noise_mse = float(
            F.mse_loss(st_noise[:, missing_index_tensor], target_noise).item()
        )
        records.append(
            {
                "fold": fold_number,
                "timestep": timestep,
                "time_fraction": timestep / (settings.diffusion_steps - 1),
                "relative_score_gap": relative_gap,
                "sc_heldout_noise_mse": sc_noise_mse,
                "st_oracle_heldout_noise_mse": st_noise_mse,
                "probe_spots": probe_count,
            }
        )

        arrow_count = min(settings.arrow_spots, probe_count)
        arrow_payload.append(
            {
                "timestep": timestep,
                "relative_score_gap": relative_gap,
                "noisy_missing": noisy[:arrow_count, missing_index_tensor].cpu().numpy(),
                "sc_score": sc_score[:arrow_count].cpu().numpy(),
                "st_score": st_score[:arrow_count].cpu().numpy(),
            }
        )

    return pd.DataFrame(records), arrow_payload


def plot_score_field(
    st_train_expression: np.ndarray,
    missing_indices: np.ndarray,
    arrow_payload: list[dict[str, np.ndarray | float | int]],
    *,
    output_path: Path,
) -> float:
    """Project paired high-dimensional conditional scores into ST-target PCA."""

    pca = PCA(n_components=2, random_state=0)
    pca.fit(st_train_expression[:, missing_indices])
    figure, axes = plt.subplots(
        1, len(arrow_payload), figsize=(5.2 * len(arrow_payload), 4.7), squeeze=False
    )

    for axis, payload in zip(axes[0], arrow_payload):
        noisy_missing = np.asarray(payload["noisy_missing"])
        sc_score = np.asarray(payload["sc_score"])
        st_score = np.asarray(payload["st_score"])
        base = pca.transform(noisy_missing)
        sc_vectors = sc_score @ pca.components_.T
        st_vectors = st_score @ pca.components_.T

        all_norms = np.concatenate(
            [np.linalg.norm(sc_vectors, axis=1), np.linalg.norm(st_vectors, axis=1)]
        )
        reference_norm = float(np.percentile(all_norms, 90)) if len(all_norms) else 1.0
        span = max(float(np.ptp(base[:, 0])), float(np.ptp(base[:, 1])), 1e-6)
        display_scale = 0.16 * span / max(reference_norm, 1e-8)
        sc_vectors = sc_vectors * display_scale
        st_vectors = st_vectors * display_scale

        axis.scatter(base[:, 0], base[:, 1], s=9, color="0.75", alpha=0.65)
        axis.quiver(
            base[:, 0], base[:, 1], sc_vectors[:, 0], sc_vectors[:, 1],
            angles="xy", scale_units="xy", scale=1, width=0.006,
            color="#D50000", alpha=0.72, label="SC score",
        )
        axis.quiver(
            base[:, 0], base[:, 1], st_vectors[:, 0], st_vectors[:, 1],
            angles="xy", scale_units="xy", scale=1, width=0.006,
            color="#0072B2", alpha=0.72, label="ST-oracle score",
        )
        axis.set_title(
            f"t = {int(payload['timestep'])}\n"
            f"relative gap = {float(payload['relative_score_gap']):.3f}"
        )
        axis.set_xlabel("ST target PC1")
        axis.set_ylabel("ST target PC2")
        axis.spines[["top", "right"]].set_visible(False)

    axes[0, 0].legend(frameon=False)
    figure.suptitle("Conditional score-field shift on identical held-out ST probes")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return float(pca.explained_variance_ratio_[:2].sum())


def plot_score_gap_summary(metrics: pd.DataFrame, output_path: Path) -> None:
    summary = (
        metrics.groupby("time_fraction")["relative_score_gap"]
        .agg(["mean", "std"])
        .reset_index()
    )
    error = summary["std"].fillna(0.0).to_numpy()
    figure, axis = plt.subplots(figsize=(6.5, 4.7))
    axis.errorbar(
        summary["time_fraction"], summary["mean"], yerr=error,
        marker="o", capsize=3, color="#6A3D9A",
    )
    axis.set_xlabel("Diffusion time fraction")
    axis.set_ylabel("Relative conditional-score gap")
    axis.set_title("SC-to-ST generative-dynamics shift")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_experiment(
    *,
    output_dir: str | Path,
    settings: ExperimentSettings,
    dataset: str | int = "dataset5",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    sc_path: str | Path | None = None,
    st_path: str | Path | None = None,
    checkpoint_run_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    allow_existing: bool = False,
) -> Path:
    """Run all requested folds and save checkpoints, metrics, and figures."""

    if (sc_path is None) != (st_path is None):
        raise ValueError("Provide both --sc-path and --st-path, or neither.")
    source_run: Path | None = None
    source_experiment: dict[str, object] | None = None
    if checkpoint_run_dir is not None:
        source_run = Path(checkpoint_run_dir).expanduser().resolve()
        experiment_path = source_run / "experiment.json"
        split_path = source_run / "data_split.json"
        if not experiment_path.is_file() or not split_path.is_file():
            raise FileNotFoundError(
                "checkpoint_run_dir must contain experiment.json and data_split.json: "
                f"{source_run}"
            )
        source_experiment = dict(load_json(experiment_path))
        source_settings = dict(source_experiment.get("settings", {}))
        required = {"n_splits", "split_seed", "model_seed", "diffusion_steps"}
        missing_settings = required - source_settings.keys()
        if missing_settings:
            raise ValueError(
                f"Prior experiment is missing settings: {sorted(missing_settings)}"
            )
        # Evaluation options remain user-controlled; structural training options
        # are inherited from the run that produced the checkpoints.
        settings = replace(
            settings,
            n_splits=int(source_settings["n_splits"]),
            split_seed=int(source_settings["split_seed"]),
            model_seed=int(source_settings["model_seed"]),
            diffusion_steps=int(source_settings["diffusion_steps"]),
        )

    if settings.epochs < 1 or settings.diffusion_steps < 2:
        raise ValueError("epochs must be positive and diffusion_steps at least two.")
    if settings.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if settings.max_mmd_samples == 1:
        raise ValueError("max_mmd_samples must be 0 or at least 2.")
    if settings.max_probe_spots == 1 or settings.arrow_spots < 1:
        raise ValueError("Use at least two probe spots and one arrow spot.")
    if not settings.time_fractions or any(
        not 0.0 < value <= 1.0 for value in settings.time_fractions
    ):
        raise ValueError("Every time fraction must lie in (0, 1].")

    run_dir = ensure_output_dir(output_dir, allow_existing=allow_existing)
    if source_run is not None and run_dir == source_run:
        raise ValueError(
            "checkpoint-only analysis requires a new output_dir; "
            "do not overwrite the training run."
        )
    resolved_device = resolve_device(device)
    if sc_path is None and source_experiment is not None:
        sc_file = Path(str(source_experiment["sc_h5ad"]))
        st_file = Path(str(source_experiment["st_h5ad"]))
        dataset_name = str(source_experiment["dataset"])
    elif sc_path is None:
        paths = resolve_dataset_paths(dataset, data_root)
        sc_file, st_file, dataset_name = paths.sc_h5ad, paths.spatial_h5ad, paths.dataset
    else:
        sc_file, st_file = Path(sc_path), Path(st_path)
        dataset_name = str(dataset)

    adata_sc, adata_st = load_paired_h5ad(sc_file, st_file)
    sc_values, st_values = preprocess_without_augmentation(adata_sc, adata_st)
    genes = np.asarray(adata_st.var_names, dtype=str)
    if source_run is None:
        sc_train_idx, st_train_idx, st_test_idx = split_and_match_samples(
            len(sc_values), len(st_values), settings
        )
    else:
        saved_split = dict(load_json(source_run / "data_split.json"))
        sc_train_idx = np.asarray(saved_split["sc_train_indices"], dtype=np.int64)
        st_train_idx = np.asarray(saved_split["st_train_indices"], dtype=np.int64)
        st_test_idx = np.asarray(saved_split["st_test_indices"], dtype=np.int64)
        if len(sc_train_idx) != len(st_train_idx):
            raise ValueError("Saved SC/ST training sample counts are not matched.")
        if (
            np.any(sc_train_idx < 0)
            or np.any(sc_train_idx >= len(sc_values))
            or np.any(st_train_idx < 0)
            or np.any(st_train_idx >= len(st_values))
            or np.any(st_test_idx < 0)
            or np.any(st_test_idx >= len(st_values))
        ):
            raise ValueError("Saved data split contains out-of-range indices.")
        if np.intersect1d(st_train_idx, st_test_idx).size:
            raise ValueError("Saved ST train/test indices overlap.")

    sc_train_values = sc_values[sc_train_idx]
    st_train_values = st_values[st_train_idx]
    st_test_values = st_values[st_test_idx]
    folds = build_gene_folds(
        genes, n_splits=settings.n_splits, random_state=settings.split_seed
    )
    if settings.fold:
        if not 1 <= settings.fold <= len(folds):
            raise ValueError(f"fold must be 0 or between 1 and {len(folds)}.")
        folds = [folds[settings.fold - 1]]

    save_json(
        {
            "mode": "checkpoint_only" if source_run is not None else "train_and_analyze",
            "checkpoint_source_run": str(source_run) if source_run else None,
            "dataset": dataset_name,
            "sc_h5ad": str(sc_file),
            "st_h5ad": str(st_file),
            "device": str(resolved_device),
            "settings": asdict(settings),
            "source_training_settings": (
                source_experiment.get("settings") if source_experiment else None
            ),
            "design": {
                "augmentation": "disabled for both models",
                "matched_training_samples": len(sc_train_idx),
                "st_test_spots": len(st_test_idx),
                "paired_initialization_timesteps_and_noise": True,
                "primary_dynamics_object": "conditional score",
                "score_relation": "score = -predicted_noise / sqrt(1-alpha_bar_t)",
            },
        },
        run_dir / "experiment.json",
    )
    save_json(
        {
            "sc_train_indices": sc_train_idx,
            "st_train_indices": st_train_idx,
            "st_test_indices": st_test_idx,
            "sc_train_obs_names": np.asarray(adata_sc.obs_names)[sc_train_idx],
            "st_train_obs_names": np.asarray(adata_st.obs_names)[st_train_idx],
            "st_test_obs_names": np.asarray(adata_st.obs_names)[st_test_idx],
        },
        run_dir / "data_split.json",
    )

    condition_records: list[dict[str, float | int]] = []
    score_frames: list[pd.DataFrame] = []
    for fold in folds:
        fold_dir = run_dir / f"fold_{fold.fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        mask = np.ones(len(genes), dtype=np.float32)
        mask[fold.test_indices] = 0.0
        observed_indices = np.flatnonzero(mask)
        missing_indices = fold.test_indices
        if len(observed_indices) < 2 or len(missing_indices) < 2:
            raise ValueError("Each fold needs at least two observed and two missing genes.")

        sc_x, sc_y, st_train_x, st_train_y = build_fold_arrays(
            sc_train_values, st_train_values, mask
        )
        st_test_x = st_test_values * 2.0 - 1.0
        st_test_y = (st_test_values * mask) * 2.0 - 1.0

        mmd_squared, bandwidth_squared = rbf_mmd_squared(
            sc_y[:, observed_indices],
            st_train_y[:, observed_indices],
            max_samples=settings.max_mmd_samples,
            seed=settings.split_seed + fold.fold,
        )
        condition_pca_variance = plot_condition_shift(
            sc_y,
            st_train_y,
            observed_indices,
            mmd_squared=mmd_squared,
            output_path=fold_dir / "conditional_distribution.png",
            max_points=settings.max_plot_points,
            seed=settings.split_seed + fold.fold,
        )
        condition_records.append(
            {
                "fold": fold.fold,
                "observed_genes": len(observed_indices),
                "missing_genes": len(missing_indices),
                "rbf_mmd_squared": mmd_squared,
                "rbf_bandwidth_squared": bandwidth_squared,
                "condition_pca_2d_variance": condition_pca_variance,
            }
        )

        if source_run is None:
            sc_model, st_model, history, model_config = train_controlled_pair(
                sc_x,
                sc_y,
                st_train_x,
                st_train_y,
                mask,
                settings=settings,
                device=resolved_device,
                fold_number=fold.fold,
            )
            history.to_csv(fold_dir / "training_history.csv", index=False)
            checkpoint_common = {
                "fold": fold.fold,
                "genes": genes.tolist(),
                "observed_indices": observed_indices,
                "missing_indices": missing_indices,
                "missing_genes": list(fold.test_genes),
                "model_config": model_config,
                "settings": asdict(settings),
            }
            torch.save(
                {
                    **checkpoint_common,
                    "source": "SC",
                    "model_state_dict": sc_model.state_dict(),
                },
                fold_dir / "sc_controlled_model.pt",
            )
            torch.save(
                {
                    **checkpoint_common,
                    "source": "ST-oracle",
                    "model_state_dict": st_model.state_dict(),
                },
                fold_dir / "st_oracle_model.pt",
            )
        else:
            sc_model, st_model, model_config, checkpoint_settings = (
                load_controlled_pair(
                    source_run,
                    fold_number=fold.fold,
                    genes=genes,
                    missing_indices=missing_indices,
                    device=resolved_device,
                )
            )
            checkpoint_steps = int(checkpoint_settings["diffusion_steps"])
            if checkpoint_steps != settings.diffusion_steps:
                raise ValueError(
                    "Checkpoint diffusion_steps does not match prior experiment metadata."
                )

        score_metrics, arrow_payload = evaluate_score_shift(
            sc_model,
            st_model,
            st_test_x,
            st_test_y,
            missing_indices,
            mask,
            settings=settings,
            device=resolved_device,
            fold_number=fold.fold,
        )
        score_metrics.to_csv(fold_dir / "score_metrics.csv", index=False)
        score_frames.append(score_metrics)
        score_pca_variance = plot_score_field(
            st_train_x,
            missing_indices,
            arrow_payload,
            output_path=fold_dir / "conditional_score_field.png",
        )
        save_json(
            {
                "fold": fold.fold,
                "observed_genes": list(fold.train_genes),
                "missing_genes": list(fold.test_genes),
                "condition_pca_2d_variance": condition_pca_variance,
                "score_pca_2d_variance": score_pca_variance,
                "checkpoint_source_fold": (
                    str(source_run / f"fold_{fold.fold:02d}") if source_run else None
                ),
            },
            fold_dir / "fold_info.json",
        )

        del sc_model, st_model
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    condition_frame = pd.DataFrame(condition_records)
    score_frame = pd.concat(score_frames, ignore_index=True)
    condition_frame.to_csv(run_dir / "condition_shift_metrics.csv", index=False)
    score_frame.to_csv(run_dir / "score_shift_metrics.csv", index=False)
    plot_score_gap_summary(score_frame, run_dir / "score_gap_summary.png")
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled SC-vs-ST stDiff assumption verification"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="dataset5")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--sc-path")
    parser.add_argument("--st-path", "--spatial-path", dest="st_path")
    parser.add_argument(
        "--checkpoint-run-dir",
        help="Prior run root; load fold checkpoints and skip all training.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0, help="0: all folds; otherwise 1-based")
    parser.add_argument("--spot-test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=STDIFF_DEFAULT_LR)
    parser.add_argument("--diffusion-steps", type=int, default=1500)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-mmd-samples", type=int, default=1000)
    parser.add_argument("--max-plot-points", type=int, default=2000)
    parser.add_argument("--max-probe-spots", type=int, default=512)
    parser.add_argument("--arrow-spots", type=int, default=40)
    parser.add_argument(
        "--time-fractions", type=float, nargs="+", default=[0.1, 0.5, 0.9]
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = ExperimentSettings(
        n_splits=args.n_splits,
        fold=args.fold,
        spot_test_fraction=args.spot_test_fraction,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        diffusion_steps=args.diffusion_steps,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        max_train_samples=args.max_train_samples,
        max_mmd_samples=args.max_mmd_samples,
        max_plot_points=args.max_plot_points,
        max_probe_spots=args.max_probe_spots,
        arrow_spots=args.arrow_spots,
        time_fractions=tuple(args.time_fractions),
        deterministic=args.deterministic,
        show_progress=not args.no_progress,
    )
    run_dir = run_experiment(
        output_dir=args.output_dir,
        settings=settings,
        dataset=args.dataset,
        data_root=args.data_root,
        sc_path=args.sc_path,
        st_path=args.st_path,
        checkpoint_run_dir=args.checkpoint_run_dir,
        device=args.device,
        allow_existing=args.allow_existing,
    )
    print(f"Assumption verification completed: {run_dir}")


if __name__ == "__main__":
    main()
