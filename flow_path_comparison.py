"""Compare the actual conditional paths of two independently trained CFM models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.linalg import solve
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

try:
    from .random_re_masking import build_masked_condition
    from .source_flow import SourceFlowODESampler, load_source_flow_checkpoint, load_source_flow_run
    from .utils import ensure_output_dir, load_json, resolve_device, save_json
except ImportError:
    from random_re_masking import build_masked_condition
    from source_flow import SourceFlowODESampler, load_source_flow_checkpoint, load_source_flow_run
    from utils import ensure_output_dir, load_json, resolve_device, save_json


@dataclass(frozen=True)
class PotentialBasis:
    """Gradients of linear, diagonal quadratic and Fourier potentials in a Euclidean chart."""

    center: np.ndarray
    bandwidth: float
    waves: np.ndarray
    directions: np.ndarray

    @classmethod
    def from_samples(cls, states: np.ndarray, waves: np.ndarray) -> PotentialBasis:
        center = states.mean(axis=0)
        bandwidth = max(float(np.sqrt(np.mean((states - center) ** 2))), 1e-6)
        axes = np.eye(center.size)
        return cls(center, bandwidth, waves, np.concatenate((axes, axes, waves, waves)))

    def factors(self, states: np.ndarray) -> np.ndarray:
        centered = (states - self.center) / self.bandwidth
        phase = centered @ self.waves.T
        return np.concatenate((np.ones_like(centered), centered, np.cos(phase), np.sin(phase)), axis=1)

    def velocity(self, states: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        return (self.factors(states) * coefficients) @ self.directions

    def fit(self, states: np.ndarray, velocity: np.ndarray, ridge: float) -> np.ndarray:
        # Weak continuity equation: E[grad psi_i · grad phi] = E[grad psi_i · dx/dt].
        # Factoring each gradient as a scalar times a direction avoids an N x K x G tensor.
        factors = self.factors(states)
        gram = (factors.T @ factors / len(states)) * (self.directions @ self.directions.T)
        rhs = (factors * (velocity @ self.directions.T)).mean(axis=0)
        gram.flat[:: len(gram) + 1] += ridge * np.diag(gram).mean()
        return solve(gram, rhs, assume_a="pos")

    def weak_residual(self, states: np.ndarray, velocity_error: np.ndarray) -> float:
        factors = self.factors(states)
        error_moment = (factors * (velocity_error @ self.directions.T)).mean(axis=0)
        rms_gradient = np.sqrt(np.mean(factors ** 2, axis=0))
        return float(np.sqrt(np.mean((error_moment / np.maximum(rms_gradient, 1e-12)) ** 2)))


def _directions(generator: np.random.Generator, count: int, dimensions: int) -> np.ndarray:
    waves = generator.standard_normal((count, dimensions))
    return waves / np.linalg.norm(waves, axis=1, keepdims=True)


@torch.no_grad()
def _path_samples(
    sampler: SourceFlowODESampler,
    condition: torch.Tensor,
    missing: torch.Tensor,
    noise: torch.Tensor,
    times: Sequence[float],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    states, velocities = [], []
    for noise_batch in noise.split(batch_size):
        batch_condition = condition.expand(noise_batch.shape[0], -1)
        path = sampler.integrate_trajectory(
            noise_batch.to(condition.device), batch_condition, (0.0, *times, 1.0)
        )[1:-1]
        # The missing marginal inherits the missing component of the actual ODE derivative.
        velocity = torch.stack([
            sampler.vector_field(state, state.new_tensor(time), batch_condition)[:, missing]
            for time, state in zip(times, path)
        ])
        states.append(path[:, :, missing].cpu().numpy())
        velocities.append(velocity.cpu().numpy())
    state_array = np.concatenate(states, axis=1).astype(np.float64)
    velocity_array = np.concatenate(velocities, axis=1).astype(np.float64)
    if not (np.isfinite(state_array).all() and np.isfinite(velocity_array).all()):
        raise FloatingPointError("A flow produced nonfinite trajectory states or velocities.")
    return state_array, velocity_array


def _test_indices(checkpoint: dict, num_spots: int) -> np.ndarray:
    partitions = []
    for key in ("train_indices", "validation_indices", "target_transfer_spot_indices"):
        indices = np.asarray(checkpoint.get(key))
        if (
            indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu"
            or indices.min() < 0 or indices.max() >= num_spots
            or np.unique(indices).size != indices.size
        ):
            raise ValueError(f"Target checkpoint has invalid ST partition {key!r}.")
        partitions.append(indices)
    joined = np.concatenate(partitions)
    if np.unique(joined).size != joined.size:
        raise ValueError("Target checkpoint ST training, validation and test spots overlap.")
    return partitions[-1]


def _load_pair(args: argparse.Namespace, device: torch.device):
    source_dir = Path(args.source_run_dir).expanduser().resolve()
    target_dir = Path(args.target_run_dir).expanduser().resolve()
    sources = {"source": source_dir, "target": target_dir}
    checkpoint_name = f"fold_{args.fold:02d}.pt"
    required = ("experiment.json", "data_info.json", "gene_folds.json", f"checkpoints/{checkpoint_name}")
    missing = [
        f"{domain}: {root / name}"
        for domain, root in sources.items() for name in required if not (root / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Train the missing flow first with source_flow_training.py "
            "(--training-domain source or target). Missing artifacts:\n" + "\n".join(missing)
        )
    experiments = {domain: load_json(root / "experiment.json") for domain, root in sources.items()}
    for domain, experiment in experiments.items():
        if experiment.get("training_domain", "source") != domain:
            raise ValueError(f"Expected a {domain}-trained flow run at {sources[domain]}.")
    source_info = load_json(source_dir / "data_info.json")
    target_info = load_json(target_dir / "data_info.json")
    if source_info["spatial_sha256"] != target_info["spatial_sha256"]:
        raise ValueError("Both runs must reference the same ST benchmark and spot order.")

    source = load_source_flow_run(
        str(source_dir), fold=args.fold, device=device, sc_path=args.sc_path,
        spatial_path=args.spatial_path, ode_steps=args.ode_steps, require_row_separable=True,
        require_observed_only_preprocessing=True,
        require_validated_performance=not args.allow_failed_performance,
    )
    target_sampler, _, target = load_source_flow_checkpoint(
        str(target_dir / "checkpoints" / checkpoint_name), device=device, num_steps=args.ode_steps,
        require_row_separable=True, training_domain="target",
        require_validated_performance=not args.allow_failed_performance,
    )
    spatial_mask = source.observed_mask.numpy()
    source_contract = source.checkpoint.get("flow_matching_contract")
    target_contract = target.get("flow_matching_contract")
    if source_contract is None:
        # Older verified source runs recorded their CFM protocol in experiment.json.
        notes = experiments["source"].get("algorithm_notes", {})
        compatible_cfm = (
            notes.get("source_distribution") == "standard_normal"
            and notes.get("probability_path") == "linear"
            and notes.get("path_schedule") == "alpha_t=t,sigma_t=1-t"
            and notes.get("objective") == "conditional_flow_matching_velocity_mse"
        )
    else:
        compatible_cfm = source_contract == target_contract
    if (
        target.get("fold") != args.fold
        or tuple(target.get("genes", ())) != source.genes
        or not (spatial_mask.any() and (~spatial_mask).any())
        or not np.array_equal(np.asarray(source.checkpoint.get("mask")), spatial_mask)
        or not np.array_equal(np.asarray(target.get("mask")), spatial_mask)
        or target.get("preprocessing") != source.checkpoint.get("preprocessing")
        or target["model_config"] != source.checkpoint["model_config"]
        or not (target_contract and compatible_cfm)
    ):
        raise ValueError("Flow checkpoints must share the fold, gene mask, preprocessing, architecture and CFM contract.")
    return source, target_sampler, target, _test_indices(target, len(source.spot_names))


def _compare_time(
    source_states: np.ndarray,
    source_velocity: np.ndarray,
    target_states: np.ndarray,
    target_velocity: np.ndarray,
    fit_samples: int,
    waves: np.ndarray,
    probe_waves: np.ndarray,
    ridge: float,
) -> dict[str, float]:
    fit_states = np.concatenate((source_states[:fit_samples], target_states[:fit_samples]))
    basis = PotentialBasis.from_samples(fit_states, waves)
    probe = PotentialBasis.from_samples(fit_states, probe_waves)
    source_coef = basis.fit(source_states[:fit_samples], source_velocity[:fit_samples], ridge)
    target_coef = basis.fit(target_states[:fit_samples], target_velocity[:fit_samples], ridge)
    source_eval = source_states[fit_samples:]
    target_eval = target_states[fit_samples:]

    # Equal-mass empirical W2 midpoint in the shared Euclidean missing-gene space.
    distances = cdist(source_eval, target_eval, metric="sqeuclidean")
    source_order, target_order = linear_sum_assignment(distances)
    midpoint = (source_eval[source_order] + target_eval[target_order]) / 2
    difference = basis.velocity(midpoint, source_coef - target_coef)
    squared_gap = float(np.mean(np.sum(difference ** 2, axis=1)))
    return {
        "squared_velocity_gap_per_gene": squared_gap / difference.shape[1],
        "half_squared_velocity_gap": squared_gap / 2,
        "w2_squared_per_gene": float(distances[source_order, target_order].mean() / difference.shape[1]),
        "source_weak_residual_rms": probe.weak_residual(
            source_eval, basis.velocity(source_eval, source_coef) - source_velocity[fit_samples:]
        ),
        "target_weak_residual_rms": probe.weak_residual(
            target_eval, basis.velocity(target_eval, target_coef) - target_velocity[fit_samples:]
        ),
    }


def run_comparison(args: argparse.Namespace) -> None:
    device = torch.device("cpu") if args.check_only else resolve_device(args.device)
    source, target_sampler, target, test_indices = _load_pair(args, device)
    print(f"Flow checkpoints ready for fold {args.fold}: {len(test_indices)} held-out ST spots.")
    if args.check_only:
        return

    output = ensure_output_dir(args.output_dir)
    random = np.random.default_rng(args.seed)
    spots = np.sort(random.choice(test_indices, size=min(args.max_spots, len(test_indices)), replace=False))
    observed = source.observed_mask.to(device)
    missing = source.missing_mask.to(device)
    conditions = build_masked_condition(source.target_expression[spots.tolist()], observed)
    dimensions = int(source.missing_mask.sum())
    waves = _directions(random, args.rff_features, dimensions)
    probe_waves = _directions(random, max(8, args.rff_features), dimensions)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    times = ((np.arange(args.num_times) + 0.5) / args.num_times).tolist()
    rows = []
    for spot, condition in zip(spots, conditions):
        noise = torch.randn(
            args.fit_samples + args.evaluation_samples, len(source.genes), generator=generator
        )
        source_states, source_velocity = _path_samples(
            source.sampler, condition, missing, noise, times, args.batch_size
        )
        target_states, target_velocity = _path_samples(
            target_sampler, condition, missing, noise, times, args.batch_size
        )
        for time_index, time in enumerate(times):
            rows.append({
                "spot_index": int(spot), "spot": source.spot_names[spot], "time": time,
                **_compare_time(
                    source_states[time_index], source_velocity[time_index],
                    target_states[time_index], target_velocity[time_index],
                    args.fit_samples, waves, probe_waves, args.ridge,
                ),
            })

    metrics = pd.DataFrame(rows)
    score_columns = metrics.columns[3:].tolist()
    metrics.to_csv(output / "path_comparison.csv", index=False)
    save_json({
        "measure": "fixed_path_minimum_kinetic_velocity_gap",
        "alignment": "identity",
        "metric": "euclidean_missing_gene_coordinates",
        "comparison_measure": "empirical_wasserstein_midpoint",
        "preprocessing": source.checkpoint["preprocessing"],
        "source_augmentation": source.checkpoint.get("augment_source", True),
        "source_validation": source.checkpoint.get("source_performance"),
        "target_validation": target.get("target_validation_performance"),
        "missing_genes": [gene for gene, keep in zip(source.genes, source.missing_mask) if keep],
        "test_spots": len(test_indices),
        "evaluated_spots": len(spots),
        "settings": vars(args),
        "mean": metrics[score_columns].mean().to_dict(),
        "by_time": metrics.groupby("time")[score_columns].mean().reset_index().to_dict("records"),
    }, output / "summary.json")
    print(f"Path comparison saved to {output}.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--target-run-dir", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--sc-path")
    parser.add_argument("--spatial-path")
    parser.add_argument("--check-only", action="store_true", help="Check the two prerequisite runs on CPU.")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--max-spots", type=int, default=16)
    parser.add_argument("--num-times", type=int, default=5)
    parser.add_argument("--fit-samples", type=int, default=128)
    parser.add_argument("--evaluation-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rff-features", type=int, default=32)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1202)
    parser.add_argument("--allow-failed-performance", action="store_true", help="Allow diagnostic use of completed checkpoints below their validation thresholds.")
    args = parser.parse_args(argv)
    if not args.check_only and not args.output_dir:
        parser.error("--output-dir is required for evaluation.")
    if min(args.fold, args.max_spots, args.num_times, args.batch_size, args.ode_steps) < 1:
        parser.error("Fold, spots, times, batch size and ODE steps must be positive.")
    if min(args.fit_samples, args.evaluation_samples) < 2 or args.rff_features < 0:
        parser.error("Fit and evaluation require at least two samples each; RFF feature count must be nonnegative.")
    if not (np.isfinite(args.ridge) and args.ridge > 0) or args.seed < 0:
        parser.error("Ridge must be positive and finite; seed must be nonnegative.")
    run_comparison(args)


if __name__ == "__main__":
    main()
