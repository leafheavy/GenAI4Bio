"""Fit and freeze shared-translation candidates for flow path comparison."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist, pdist

try:
    from .metric import fixed_rbf_mmd_squared
    from .utils import ensure_output_dir, load_json, save_json, sha256_file
except ImportError:
    from metric import fixed_rbf_mmd_squared
    from utils import ensure_output_dir, load_json, save_json, sha256_file


@dataclass(frozen=True)
class SharedTranslation:
    delta: float

    def forward_condition(self, source_observed: torch.Tensor) -> torch.Tensor:
        return source_observed + self.delta

    def inverse_condition(self, target_observed: torch.Tensor) -> torch.Tensor:
        return target_observed - self.delta

    def push_forward(
        self, source_state: torch.Tensor, source_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return source_state + self.delta, source_velocity


def _fit_delta(source: np.ndarray, target: np.ndarray, bandwidth: float) -> float:
    # Translating the source leaves its within-domain kernel unchanged.
    # Minimize MMD by maximizing the cross-domain kernel mean.
    distance_squared = cdist(source, target, metric="sqeuclidean")
    cross_term = 2 * (source.sum(axis=1)[:, None] - target.sum(axis=1)[None, :])

    def objective(delta: float) -> float:
        shifted = distance_squared + delta * (cross_term + source.shape[1] * delta)
        return -float(np.exp(-np.maximum(shifted, 0) / (2 * bandwidth**2)).mean())

    # For source and target coordinates in [-1, 1], all cross-kernel
    # terms peak at offsets in [-2, 2]. Refine the best grid interval.
    grid = np.linspace(-2.0, 2.0, 81)
    index = min(range(len(grid)), key=lambda i: objective(grid[i]))
    bounds = (grid[max(0, index - 1)], grid[min(len(grid) - 1, index + 1)])
    result = minimize_scalar(objective, method="bounded", bounds=bounds)
    return float(result.x if result.success and result.fun < objective(grid[index]) else grid[index])


def prepare(args: argparse.Namespace) -> Path:
    try:
        from .Dataloader import GENE_LOCAL_PREPROCESSING, read_h5ad, scale_gene_local
        from .flow_path_comparison import _target_partitions
    except ImportError:
        from Dataloader import GENE_LOCAL_PREPROCESSING, read_h5ad, scale_gene_local
        from flow_path_comparison import _target_partitions

    source_run = Path(args.source_run_dir).expanduser()
    target_run = Path(args.target_run_dir).expanduser()
    source_info = load_json(source_run / "data_info.json")
    target_info = load_json(target_run / "data_info.json")
    hashes = {key: source_info[key] for key in ("sc_sha256", "spatial_sha256")}
    if target_info["spatial_sha256"] != hashes["spatial_sha256"]:
        raise ValueError("Source and target runs must share the ST dataset.")

    checkpoint_name = f"checkpoints/fold_{args.fold:02d}.pt"
    source = torch.load(source_run / checkpoint_name, map_location="cpu", weights_only=False)
    target = torch.load(target_run / checkpoint_name, map_location="cpu", weights_only=False)
    genes = tuple(source["genes"])
    mask = np.asarray(source["mask"], dtype=bool)
    if tuple(source_info["genes"]) != genes or mask.shape != (len(genes),) or not mask.any() or mask.all():
        raise ValueError("The source checkpoint must contain the expected genes and a partial mask.")
    for domain, checkpoint in (("source", source), ("target", target)):
        if (
            checkpoint.get("method") != "source_flow"
            or checkpoint.get("training_domain", "source") != domain
            or checkpoint.get("training_status", "complete") != "complete"
            or checkpoint.get("fold") != args.fold
            or checkpoint.get("preprocessing") != GENE_LOCAL_PREPROCESSING
            or tuple(checkpoint["genes"]) != genes
            or not np.array_equal(checkpoint["mask"], mask)
        ):
            raise ValueError(f"Incompatible {domain} checkpoint for observed calibration.")
    del source

    sc_path = Path(args.sc_path or source_info["sc_h5ad"]).expanduser()
    spatial_path = Path(args.spatial_path or source_info["spatial_h5ad"]).expanduser()
    if sha256_file(sc_path) != hashes["sc_sha256"] or sha256_file(spatial_path) != hashes["spatial_sha256"]:
        raise ValueError("H5AD files differ from the flow training inputs.")
    sc = read_h5ad(sc_path)
    st = read_h5ad(spatial_path)
    if tuple(sc.var_names) != genes or tuple(st.var_names) != genes:
        raise ValueError("H5AD gene order differs from the checkpoint.")
    target_train, target_validation, target_test = _target_partitions(target, st.n_obs)
    del target, checkpoint

    rng = np.random.default_rng(args.seed)
    if sc.n_obs < 5:
        raise ValueError("At least five original source rows are required for calibration splits.")
    heldout_count = max(1, round(0.2 * sc.n_obs))
    source_test, source_validation, source_train = np.split(
        rng.permutation(sc.n_obs), [heldout_count, 2 * heldout_count]
    )

    def subset(rows: np.ndarray) -> np.ndarray:
        return np.sort(rng.choice(rows, size=min(args.max_fit_rows, len(rows)), replace=False))

    def observed(adata, rows: np.ndarray) -> np.ndarray:
        matrix = adata[rows, mask].copy()
        raw = matrix.X.data if hasattr(matrix.X, "toarray") else matrix.X
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError("Raw observed expression must be finite and nonnegative.")
        scale_gene_local(matrix)
        values = matrix.X.toarray() if hasattr(matrix.X, "toarray") else matrix.X
        return np.asarray(values, dtype=np.float64) * 2 - 1

    source_fit = observed(sc, subset(source_train))
    target_fit = observed(st, subset(target_train))
    source_val = observed(sc, subset(source_validation))
    target_val = observed(st, subset(target_validation))
    distances = pdist(np.concatenate((source_fit, target_fit)))
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError("Training observations have no positive pairwise distance for RBF bandwidth.")
    bandwidth = float(np.median(positive))
    train_delta = _fit_delta(source_fit, target_fit, bandwidth)

    def validation_mmd(delta: float) -> float:
        return fixed_rbf_mmd_squared(source_val + delta, target_val, bandwidth=bandwidth)

    # Select on observed validation data, then freeze the perturbation grid.
    offsets = (-0.10, -0.05, 0.0, 0.05, 0.10)
    selected = min((0.0, *(train_delta + offset for offset in offsets)), key=validation_mmd)
    candidates = []
    for index, offset in enumerate(offsets, start=1):
        delta = selected + offset
        if abs(delta) < 1e-8:
            continue  # The rank evaluator already computes the identity baseline.
        inverse = target_val - delta
        candidates.append({
            "name": f"translation_{index:02d}",
            "delta": delta,
            "validation_mmd_squared": validation_mmd(delta),
            "validation_inverse_outside_unit_range_fraction": float(
                np.mean((inverse < -1) | (inverse > 1))
            ),
        })

    output = ensure_output_dir(args.output_dir)
    np.save(output / "source_observed_test.npy", observed(sc, np.sort(source_test)).astype(np.float32))
    save_json({
        "method": "shared_translation_v1",
        "fold": args.fold,
        "preprocessing": GENE_LOCAL_PREPROCESSING,
        "observed_genes": [gene for gene, keep in zip(genes, mask) if keep],
        **hashes,
        "source_test_indices": np.sort(source_test).tolist(),
        "target_test_indices": np.sort(target_test).tolist(),
        "seed": args.seed,
        "max_fit_rows": args.max_fit_rows,
        "mmd_bandwidth": bandwidth,
        "train_delta": train_delta,
        "selected_delta": selected,
        "baseline_validation_mmd_squared": validation_mmd(0.0),
        "candidates": candidates,
    }, output / "calibration.json")
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--target-run-dir", required=True, help="One ST reference run with the shared split.")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--sc-path")
    parser.add_argument("--spatial-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-fit-rows", type=int, default=512, help="Per-domain cap for fit and validation.")
    parser.add_argument("--seed", type=int, default=1212)
    args = parser.parse_args(argv)
    if args.fold < 1 or args.max_fit_rows < 2 or args.seed < 0:
        parser.error("Fold must be positive, max-fit-rows at least 2, and seed nonnegative.")
    print(prepare(args))


if __name__ == "__main__":
    main()
