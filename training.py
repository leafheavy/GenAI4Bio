"""Training entry point shared by stDiff and IMD-T."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from Dataloader import (
    DEFAULT_DATA_ROOT,
    GENE_LOCAL_PREPROCESSING,
    LEGACY_PREPROCESSING,
    PreparedData,
    build_fold_data,
    build_gene_folds,
    fold_records,
    get_data_loader,
    load_paired_h5ad,
    preprocess_pair,
    resolve_dataset_paths,
)
from model import (
    BATCH_ATTENTION,
    GENE_ATTENTION,
    build_model,
    build_noise_scheduler,
    resolve_attention_architecture,
)
from utils import (
    ensure_output_dir,
    environment_snapshot,
    load_json,
    resolve_device,
    save_json,
    set_seed,
    sha256_file,
)


STDIFF_DEFAULT_LR = 0.00016046744893538737
IMDT_DEFAULT_LR = 1e-6


@dataclass(frozen=True)
class TrainSettings:
    batch_size: int = 512
    hidden_size: int = 1024
    depth: int = 6
    num_heads: int = 16
    mlp_ratio: float = 4.0
    diffusion_steps: int = 1500
    epochs: int = 900
    learning_rate: float | None = None
    noise_std: float = 10.0
    n_splits: int = 5
    split_seed: int = 0
    model_seed: int = 1202
    leiden_resolution: float = 0.4
    num_workers: int = 0
    show_progress: bool = True
    deterministic: bool = False
    make_plots: bool = True

    @classmethod
    def from_mapping(
        cls,
        method: str,
        values: Mapping[str, Any] | None,
        *,
        architecture: str | None = None,
    ) -> "TrainSettings":
        architecture = resolve_attention_architecture(method, architecture)
        payload = dict(values or {})
        if payload.get("batch_size") is None:
            payload["batch_size"] = 8 if architecture == GENE_ATTENTION else cls.batch_size
        if payload.get("hidden_size") is None:
            payload["hidden_size"] = 256 if architecture == GENE_ATTENTION else cls.hidden_size
        if payload.get("learning_rate") is None:
            payload["learning_rate"] = (
                STDIFF_DEFAULT_LR
                if method.lower() == "stdiff"
                else IMDT_DEFAULT_LR
            )
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise TypeError(f"Unknown training parameters: {sorted(unknown)}")
        settings = cls(**payload)
        if settings.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2.")
        if settings.epochs < 1:
            raise ValueError("epochs must be positive.")
        if settings.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        return settings


def _plot_training_history(history: pd.DataFrame, *, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    required = {"epoch", "loss"}
    if not required.issubset(history.columns):
        raise ValueError(f"Training history must contain {sorted(required)}.")
    fig, ax = plt.subplots(figsize=(8, 5))
    if "fold" in history:
        for fold, frame in history.groupby("fold", sort=True):
            ax.plot(frame["epoch"], frame["loss"], label=f"fold {fold}")
        ax.legend(frameon=False, ncol=2)
    else:
        ax.plot(history["epoch"], history["loss"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Noise-prediction loss")
    ax.set_title("Training history")
    sns.despine(ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_one_fold(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    method: str,
    mask: np.ndarray,
    learning_rate: float,
    epochs: int,
    diffusion_steps: int,
    device: torch.device,
    show_progress: bool = True,
) -> tuple[list[dict[str, float | int]], torch.optim.Optimizer]:
    """Run the original method-specific noise-prediction objective."""

    method_key = method.lower()
    if method_key not in {"stdiff", "imdt"}:
        raise ValueError("method must be 'stdiff' or 'imdt'.")

    noise_scheduler = build_noise_scheduler(method_key, diffusion_steps)
    criterion = nn.MSELoss()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0
    )
    mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=device)
    history: list[dict[str, float | int]] = []
    epoch_iterator = (
        tqdm(range(epochs), ncols=100, desc=f"train {method_key}")
        if show_progress
        else range(epochs)
    )
    model.train()

    for epoch in epoch_iterator:
        epoch_loss = 0.0
        batch_count = 0
        for expression, condition in dataloader:
            expression = expression.float().to(device)
            if method_key == "stdiff":
                condition = condition.float().to(device)
            else:
                condition = condition.long().to(device)

            noise = torch.randn_like(expression)
            # Scheduler coefficients stay on CPU, matching the prior work.
            timesteps = torch.randint(
                1, diffusion_steps, (expression.shape[0],), device="cpu"
            ).long()
            noisy = noise_scheduler.add_noise(expression, noise, timesteps)

            if method_key == "stdiff":
                model_input = noisy * (1 - mask_tensor) + expression * mask_tensor
            else:
                # IMD-T intentionally trains on all genes; fold mask is not in its loss.
                model_input = noisy

            optimizer.zero_grad(set_to_none=True)
            noise_prediction = model(
                model_input,
                t=timesteps.to(device),
                y=condition,
            )
            if method_key == "stdiff":
                loss = criterion(
                    noise * (1 - mask_tensor),
                    noise_prediction * (1 - mask_tensor),
                )
            else:
                loss = criterion(noise_prediction, noise)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            batch_count += 1

        if batch_count == 0:
            raise RuntimeError("The training DataLoader produced no batches.")
        epoch_loss /= batch_count
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        if show_progress:
            epoch_iterator.set_postfix_str(f"noise loss:{epoch_loss:.5f}")

    return history, optimizer


def _resolve_input_paths(
    dataset: str | int,
    data_root: str | Path,
    sc_path: str | Path | None,
    spatial_path: str | Path | None,
) -> tuple[Path, Path, str]:
    if (sc_path is None) != (spatial_path is None):
        raise ValueError("sc_path and spatial_path must be provided together.")
    resolved = resolve_dataset_paths(dataset, data_root)
    if sc_path is not None and spatial_path is not None:
        return Path(sc_path), Path(spatial_path), resolved.dataset
    return resolved.sc_h5ad, resolved.spatial_h5ad, resolved.dataset


def _imdt_classes(prepared: PreparedData) -> int:
    if prepared.seq_labels is None or prepared.spatial_labels is None:
        raise ValueError("IMD-T preprocessing did not produce Leiden labels.")
    classes = int(len(np.unique(prepared.seq_labels)))
    if prepared.spatial_labels.size and int(prepared.spatial_labels.max()) >= classes:
        raise ValueError(
            "Spatial Leiden code exceeds the scRNA embedding range: "
            f"max spatial code={int(prepared.spatial_labels.max())}, classes={classes}."
        )
    return classes


def train_experiment(
    method: str,
    *,
    output_dir: str | Path,
    dataset: str | int = "dataset5",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    sc_path: str | Path | None = None,
    spatial_path: str | Path | None = None,
    seed: int = 0,
    train_params: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
    allow_existing: bool = False,
    architecture: str | None = None,
) -> Path:
    """Train all gene folds and save portable, structured checkpoints."""

    method_key = method.strip().lower()
    if method_key not in {"stdiff", "imdt"}:
        raise ValueError("method must be 'stdiff' or 'imdt'.")
    architecture = resolve_attention_architecture(method_key, architecture)
    settings_values = dict(train_params or {})
    settings_values.setdefault("split_seed", seed)
    settings = TrainSettings.from_mapping(
        method_key, settings_values, architecture=architecture
    )
    resolved_device = resolve_device(device)
    run_dir = ensure_output_dir(output_dir, allow_existing=allow_existing)
    existing_experiment_path = run_dir / "experiment.json"
    if existing_experiment_path.is_file():
        existing = load_json(existing_experiment_path)
        if existing.get("preprocessing", LEGACY_PREPROCESSING) != GENE_LOCAL_PREPROCESSING:
            raise ValueError("Existing run uses legacy preprocessing; choose a new output directory.")
        existing_architecture = existing.get("model", {}).get("architecture")
        if existing_architecture is None and existing.get("method") == "imdt":
            existing_architecture = BATCH_ATTENTION
        if existing.get("method") != method_key or existing_architecture != architecture:
            raise ValueError(
                "Existing run has different or unrecorded model architecture; "
                "choose a new output directory."
            )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sc_file, spatial_file, dataset_name = _resolve_input_paths(
        dataset, data_root, sc_path, spatial_path
    )
    adata_seq, adata_spatial = load_paired_h5ad(
        sc_file,
        spatial_file,
        require_spatial_coords=(method_key == "imdt"),
    )
    prepared = preprocess_pair(
        adata_seq,
        adata_spatial,
        method=method_key,
        noise_std=settings.noise_std,
        leiden_resolution=settings.leiden_resolution,
        leiden_random_state=settings.split_seed,
        preprocessing=GENE_LOCAL_PREPROCESSING,
    )
    folds = build_gene_folds(
        prepared.genes,
        n_splits=settings.n_splits,
        random_state=settings.split_seed,
    )
    classes = _imdt_classes(prepared) if method_key == "imdt" else 6

    data_info = {
        "dataset": dataset_name,
        "sc_h5ad": str(sc_file),
        "spatial_h5ad": str(spatial_file),
        "sc_sha256": sha256_file(sc_file),
        "spatial_sha256": sha256_file(spatial_file),
        "sc_shape_raw": list(adata_seq.shape),
        "spatial_shape_raw": list(adata_spatial.shape),
        "sc_shape_after_augmentation": list(prepared.seq_array.shape),
        "genes": prepared.genes.tolist(),
    }
    if method_key == "imdt":
        data_info["sc_leiden_classes"] = classes
        data_info["spatial_leiden_classes"] = int(
            len(np.unique(prepared.spatial_labels))
        )
    save_json(data_info, run_dir / "data_info.json")
    save_json(fold_records(folds), run_dir / "gene_folds.json")

    experiment = {
        "method": method_key,
        "preprocessing": GENE_LOCAL_PREPROCESSING,
        "dataset": dataset_name,
        "output_dir": str(run_dir),
        "device": str(resolved_device),
        "settings": asdict(settings),
        "model": {
            "architecture": architecture,
            "input_size": int(prepared.spatial_array.shape[1]),
            "hidden_size": settings.hidden_size,
            "depth": settings.depth,
            "num_heads": settings.num_heads,
            "classes": classes,
            "mlp_ratio": settings.mlp_ratio,
        },
        "environment": environment_snapshot(),
        "algorithm_notes": {
            "stdiff_masked_loss": method_key == "stdiff",
            "imdt_full_gene_loss": method_key == "imdt",
            "imdt_default_repaint": False if method_key == "imdt" else None,
        },
    }
    save_json(experiment, run_dir / "experiment.json")

    all_history: list[dict[str, float | int | str]] = []
    for fold in folds:
        # The original scripts reset model/CUDA seeds to 1202 for every fold.
        set_seed(settings.model_seed, deterministic=settings.deterministic)
        fold_data = build_fold_data(prepared, fold, method=method_key)
        train_loader = get_data_loader(
            fold_data.train_expression,
            fold_data.train_condition,
            batch_size=settings.batch_size,
            is_shuffle=True,
            seed=settings.model_seed,
            num_workers=settings.num_workers,
        )
        model = build_model(
            method_key,
            input_size=prepared.spatial_array.shape[1],
            hidden_size=settings.hidden_size,
            depth=settings.depth,
            num_heads=settings.num_heads,
            classes=classes,
            mlp_ratio=settings.mlp_ratio,
            architecture=architecture,
        )
        history, optimizer = train_one_fold(
            model,
            train_loader,
            method=method_key,
            mask=fold_data.mask,
            learning_rate=float(settings.learning_rate),
            epochs=settings.epochs,
            diffusion_steps=settings.diffusion_steps,
            device=resolved_device,
            show_progress=settings.show_progress,
        )
        for record in history:
            all_history.append(
                {"method": method_key, "fold": fold.fold, **record}
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "method": method_key,
            "fold": fold.fold,
            "test_indices": fold.test_indices,
            "test_genes": list(fold.test_genes),
            "genes": prepared.genes.tolist(),
            "mask": fold_data.mask,
            "model_config": experiment["model"],
            "training_settings": asdict(settings),
            "epoch": settings.epochs,
        }
        torch.save(checkpoint, checkpoint_dir / f"fold_{fold.fold:02d}.pt")

    history_frame = pd.DataFrame(all_history)
    history_frame.to_csv(run_dir / "train_history.csv", index=False)
    if settings.make_plots and not history_frame.empty:
        _plot_training_history(
            history_frame,
            output_path=run_dir / "training_loss.png",
        )
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train migrated stDiff or IMD-T")
    parser.add_argument("--method", choices=("stdiff", "imdt"), required=True)
    parser.add_argument(
        "--architecture",
        choices=(GENE_ATTENTION, BATCH_ATTENTION),
        help="Default: gene_attention for stDiff, batch_attention for IMD-T.",
    )
    parser.add_argument("--dataset", default="dataset5")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--sc-path", default=None)
    parser.add_argument("--spatial-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--batch-size", type=int, help="Default: 8 for gene attention, 512 otherwise."
    )
    parser.add_argument(
        "--hidden-size", type=int, help="Default: 256 for gene attention, 1024 otherwise."
    )
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--noise-std", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    train_experiment(
        method=args.method,
        dataset=args.dataset,
        data_root=args.data_root,
        sc_path=args.sc_path,
        spatial_path=args.spatial_path,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        allow_existing=args.allow_existing,
        architecture=args.architecture,
        train_params={
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "num_heads": args.heads,
            "diffusion_steps": args.steps,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "noise_std": args.noise_std,
            "show_progress": not args.no_progress,
            "make_plots": not args.no_plots,
        },
    )


if __name__ == "__main__":
    main()


__all__ = [
    "IMDT_DEFAULT_LR",
    "STDIFF_DEFAULT_LR",
    "TrainSettings",
    "train_experiment",
    "train_one_fold",
]
