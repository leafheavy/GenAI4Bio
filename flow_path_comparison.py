"""Compare the actual conditional paths of two independently trained CFM models."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

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


class CalibrationAdapter(Protocol):
    """Supply inverse condition and state/tangent maps for a fitted calibration.

    Conditions contain only observed genes. The forward map receives batches
    of full source fibre states and their tangents, with zero observed velocity.
    It returns full target states and the differential applied to the tangents.
    """

    def inverse_condition(self, target_observed: torch.Tensor) -> torch.Tensor: ...

    def push_forward(
        self, source_state: torch.Tensor, source_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class ObservedCalibrationAdapter(CalibrationAdapter, Protocol):
    """A fitted calibration that can also transform batches of observed source rows."""

    def forward_condition(self, source_observed: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class CalibrationRankEvaluation:
    """Compare fixed calibrations using the same held-out source observations."""

    candidates: Mapping[str, ObservedCalibrationAdapter]
    source_observed: np.ndarray | torch.Tensor
    mmd_bandwidth: float


@dataclass(frozen=True)
class VelocitySlice:
    midpoint: np.ndarray
    source: np.ndarray
    target: np.ndarray


def _load_prepared_calibration(
    args: argparse.Namespace, source, test_indices: np.ndarray
) -> CalibrationRankEvaluation:
    try:
        from .prepare_observed_calibration import SharedTranslation
    except ImportError:
        from prepare_observed_calibration import SharedTranslation

    root = Path(args.calibration_dir).expanduser()
    metadata = load_json(root / "calibration.json")
    source_info = load_json(Path(args.source_run_dir).expanduser().resolve() / "data_info.json")
    observed_genes = [gene for gene, keep in zip(source.genes, source.observed_mask) if keep]
    if (
        metadata.get("method") != "shared_translation_v1"
        or metadata.get("fold") != args.fold
        or metadata.get("observed_genes") != observed_genes
        or metadata.get("sc_sha256") != source_info["sc_sha256"]
        or metadata.get("spatial_sha256") != source_info["spatial_sha256"]
        or metadata.get("preprocessing") != source.checkpoint["preprocessing"]
        or metadata.get("target_test_indices") != np.sort(test_indices).tolist()
    ):
        raise ValueError("Prepared calibration differs from the evaluation dataset, fold or held-out split.")
    observed = np.load(root / "source_observed_test.npy", allow_pickle=False)
    candidates = {item["name"]: SharedTranslation(float(item["delta"])) for item in metadata["candidates"]}
    bandwidth = float(metadata["mmd_bandwidth"])
    if (
        not candidates or not all(np.isfinite(adapter.delta) for adapter in candidates.values())
        or not np.isfinite(bandwidth) or bandwidth <= 0
        or observed.shape != (len(metadata["source_test_indices"]), len(observed_genes))
        or not np.isfinite(observed).all()
    ):
        raise ValueError("Invalid candidates, bandwidth or source observations in prepared calibration.")
    return CalibrationRankEvaluation(
        candidates=candidates,
        source_observed=observed,
        mmd_bandwidth=bandwidth,
    )


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
    *,
    calibration: CalibrationAdapter | None = None,
    include_endpoint: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    if calibration is not None:
        observed = ~missing
        target_observed = condition[observed]
        source_observed = calibration.inverse_condition(target_observed.clone())
        if source_observed.shape != target_observed.shape or not torch.isfinite(source_observed).all():
            raise ValueError("Inverse calibration must return a finite observed condition of the same shape.")
        condition = condition.clone()
        condition[observed] = source_observed

    states, velocities, endpoints = [], [], []
    for noise_batch in noise.split(batch_size):
        batch_condition = condition.expand(noise_batch.shape[0], -1)
        path = sampler.integrate_trajectory(
            noise_batch.to(condition.device), batch_condition, (0.0, *times, 1.0)
        )[1:]
        # The missing marginal inherits the missing component of the actual ODE derivative.
        velocity = torch.stack([
            sampler.vector_field(state, state.new_tensor(time), batch_condition)[:, missing]
            for time, state in zip(times, path[:-1])
        ])
        if not include_endpoint:
            path = path[:-1]
        if calibration is not None:
            # The sampler evolves the full state, but the compared path lives
            # on the conditional fibre with the observed coordinates fixed.
            source_state = path.reshape(-1, path.shape[-1]).clone()
            source_state[:, observed] = source_observed
            source_velocity = torch.zeros_like(path)
            source_velocity[:len(times), :, missing] = velocity
            flat_velocity = source_velocity.reshape_as(source_state)
            target_state, target_velocity = calibration.push_forward(source_state, flat_velocity)
            if target_state.shape != source_state.shape or target_velocity.shape != flat_velocity.shape:
                raise ValueError("Calibration must preserve the full state and tangent shapes.")
            observed_states = target_state[:, observed]
            observed_velocities = target_velocity[:, observed]
            if not (
                torch.allclose(observed_states, target_observed.expand_as(observed_states), atol=1e-6)
                and torch.allclose(observed_velocities, torch.zeros_like(observed_velocities), atol=1e-6)
            ):
                raise ValueError("Calibration must send source states and tangents to the target condition fibre.")
            path = target_state.reshape_as(path)
            velocity = target_velocity.reshape_as(path)[:len(times), :, missing]
        if include_endpoint:
            endpoints.append(path[-1, :, missing].cpu().numpy())
            path = path[:-1]
        states.append(path[:, :, missing].cpu().numpy())
        velocities.append(velocity.cpu().numpy())
    state_array = np.concatenate(states, axis=1).astype(np.float64)
    velocity_array = np.concatenate(velocities, axis=1).astype(np.float64)
    if not (np.isfinite(state_array).all() and np.isfinite(velocity_array).all()):
        raise FloatingPointError("A flow produced nonfinite trajectory states or velocities.")
    if not include_endpoint:
        return state_array, velocity_array
    endpoint_array = np.concatenate(endpoints).astype(np.float64)
    if not np.isfinite(endpoint_array).all():
        raise FloatingPointError("A flow produced nonfinite endpoint states.")
    return state_array, velocity_array, endpoint_array


def _target_partitions(checkpoint: dict, num_spots: int) -> list[np.ndarray]:
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
    return partitions


def _load_pairs(args: argparse.Namespace, device: torch.device):
    source_dir = Path(args.source_run_dir).expanduser().resolve()
    target_runs = args.target_run_dir
    if isinstance(target_runs, (str, Path)):
        target_runs = [target_runs]
    target_dirs = [Path(value).expanduser().resolve() for value in target_runs]
    if not target_dirs or len(set(target_dirs)) != len(target_dirs):
        raise ValueError("Provide at least one target run and list each target directory once.")
    sources = [("source", source_dir), *(("target", root) for root in target_dirs)]
    checkpoint_name = f"fold_{args.fold:02d}.pt"
    required = ("experiment.json", "data_info.json", "gene_folds.json", f"checkpoints/{checkpoint_name}")
    missing = [
        f"{domain}: {root / name}"
        for domain, root in sources for name in required if not (root / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Train the missing flow first with source_flow_training.py "
            "(--training-domain source or target). Missing artifacts:\n" + "\n".join(missing)
        )
    for domain, root in sources:
        experiment = load_json(root / "experiment.json")
        if experiment.get("training_domain", "source") != domain:
            raise ValueError(f"Expected a {domain}-trained flow run at {root}.")
    source_experiment = load_json(source_dir / "experiment.json")
    source_info = load_json(source_dir / "data_info.json")

    source = load_source_flow_run(
        str(source_dir), fold=args.fold, device=device, sc_path=args.sc_path,
        spatial_path=args.spatial_path, ode_steps=args.ode_steps, require_row_separable=True,
        require_observed_only_preprocessing=True,
        require_validated_performance=not args.allow_failed_performance,
    )
    spatial_mask = source.observed_mask.numpy()
    source_contract = source.checkpoint.get("flow_matching_contract")
    notes = source_experiment.get("algorithm_notes", {})
    legacy_cfm = (
        notes.get("source_distribution") == "standard_normal"
        and notes.get("probability_path") == "linear"
        and notes.get("path_schedule") == "alpha_t=t,sigma_t=1-t"
        and notes.get("objective") == "conditional_flow_matching_velocity_mse"
    )
    reference_partitions = None
    target_seeds: set[int] = set()
    for target_dir in target_dirs:
        target_info = load_json(target_dir / "data_info.json")
        if source_info["spatial_sha256"] != target_info["spatial_sha256"]:
            raise ValueError("All runs must reference the same ST benchmark and spot order.")
        target_sampler, _, target = load_source_flow_checkpoint(
            str(target_dir / "checkpoints" / checkpoint_name), device=device, num_steps=args.ode_steps,
            require_row_separable=True, training_domain="target",
            require_validated_performance=not args.allow_failed_performance,
        )
        target_contract = target.get("flow_matching_contract")
        compatible_cfm = legacy_cfm if source_contract is None else source_contract == target_contract
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
        seed = target.get("training_settings", {}).get("model_seed")
        if not isinstance(seed, int) or seed in target_seeds:
            raise ValueError(f"Target runs must have distinct training model seeds: {target_dir}.")
        target_seeds.add(seed)
        partitions = _target_partitions(target, len(source.spot_names))
        if reference_partitions is None:
            reference_partitions = partitions
        elif any(
            not np.array_equal(np.sort(indices), np.sort(reference))
            for indices, reference in zip(partitions, reference_partitions)
        ):
            raise ValueError("Target runs must share the same ST training, validation and test partitions.")
        yield source, target_sampler, target, reference_partitions[-1], target_dir, seed
        del target_sampler, target, _


def _compare_time_and_candidates(
    source_states: np.ndarray,
    source_velocity: np.ndarray,
    target_states: np.ndarray,
    target_velocity: np.ndarray,
    fit_samples: int,
    waves: np.ndarray,
    probe_waves: np.ndarray,
    ridge: float,
    *,
    calibrated_source: tuple[np.ndarray, np.ndarray] | None = None,
    rank_sources: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    capture_velocity: bool = False,
) -> tuple[dict[str, float], dict[str, float], VelocitySlice | None]:
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
    scores = {
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
    if calibrated_source is not None:
        calibrated_states, calibrated_velocity = calibrated_source
        calibrated_coef = basis.fit(
            calibrated_states[:fit_samples], calibrated_velocity[:fit_samples], ridge
        )
        calibrated_difference = basis.velocity(midpoint, calibrated_coef - target_coef)
        calibrated_gap = float(np.mean(np.sum(calibrated_difference ** 2, axis=1))) / 2
        calibrated_eval = calibrated_states[fit_samples:]
        scores.update(
            calibrated_half_squared_velocity_gap=calibrated_gap,
            half_squared_velocity_gap_reduction=squared_gap / 2 - calibrated_gap,
            calibrated_source_weak_residual_rms=probe.weak_residual(
                calibrated_eval,
                basis.velocity(calibrated_eval, calibrated_coef) - calibrated_velocity[fit_samples:],
            ),
        )
    candidate_losses = {}
    for name, (states, velocities) in (rank_sources or {}).items():
        coefficients = basis.fit(states[:fit_samples], velocities[:fit_samples], ridge)
        difference = basis.velocity(midpoint, coefficients - target_coef)
        candidate_losses[name] = float(np.mean(np.sum(difference ** 2, axis=1))) / 2
    velocity_slice = None
    if capture_velocity:
        velocity_slice = VelocitySlice(
            midpoint, basis.velocity(midpoint, source_coef), basis.velocity(midpoint, target_coef)
        )
    return scores, candidate_losses, velocity_slice


def _compare_time(
    source_states: np.ndarray,
    source_velocity: np.ndarray,
    target_states: np.ndarray,
    target_velocity: np.ndarray,
    fit_samples: int,
    waves: np.ndarray,
    probe_waves: np.ndarray,
    ridge: float,
    *,
    calibrated_source: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    scores, _, _ = _compare_time_and_candidates(
        source_states, source_velocity, target_states, target_velocity, fit_samples,
        waves, probe_waves, ridge, calibrated_source=calibrated_source,
    )
    return scores


def _endpoint_mse(samples: np.ndarray, truth: np.ndarray, fit_samples: int) -> float:
    prediction = np.clip((samples[fit_samples:].mean(axis=0) + 1.0) / 2.0, 0.0, 1.0)
    return float(np.mean((prediction - truth) ** 2))


def _compare_seed(
    args: argparse.Namespace,
    source,
    target_sampler: SourceFlowODESampler,
    test_indices: np.ndarray,
    target_seed: int,
    calibration: CalibrationAdapter | None,
    rank_candidates: Mapping[str, ObservedCalibrationAdapter],
    score_endpoints: bool,
    *,
    velocity_slices: dict[float, VelocitySlice] | None = None,
) -> tuple[list[dict[str, object]], dict[str, float], tuple[float, dict[str, float]] | None]:
    random = np.random.default_rng(args.seed)
    spots = np.sort(random.choice(test_indices, size=min(args.max_spots, len(test_indices)), replace=False))
    device = source.target_expression.device
    observed = source.observed_mask.to(device)
    missing = source.missing_mask.to(device)
    conditions = build_masked_condition(source.target_expression[spots.tolist()], observed)
    dimensions = int(source.missing_mask.sum())
    if velocity_slices is not None and dimensions < 2:
        raise ValueError("The velocity plot requires at least two missing genes for PCA.")
    waves = _directions(random, args.rff_features, dimensions)
    probe_waves = _directions(random, max(8, args.rff_features), dimensions)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    times = ((np.arange(args.num_times) + 0.5) / args.num_times).tolist()
    plot_spot = None
    if velocity_slices is not None:
        plot_spot = int(spots[0]) if args.velocity_spot is None else args.velocity_spot
        if plot_spot not in spots:
            raise ValueError(f"Velocity plot spot {plot_spot} is not among the sampled ST test spots.")
    plot_times = {0, len(times) // 2, len(times) - 1}
    rows = []
    candidate_path_loss = dict.fromkeys(rank_candidates, 0.0)
    baseline_mse = 0.0
    candidate_mse = dict.fromkeys(rank_candidates, 0.0)
    for spot, condition in zip(spots, conditions):
        noise = torch.randn(
            args.fit_samples + args.evaluation_samples, len(source.genes), generator=generator
        )
        if velocity_slices is not None and spot != plot_spot:
            # Advance the same noise stream as the numeric comparison.
            continue
        source_samples = _path_samples(
            source.sampler, condition, missing, noise, times, args.batch_size,
            include_endpoint=score_endpoints,
        )
        source_states, source_velocity = source_samples[:2]
        target_states, target_velocity = _path_samples(
            target_sampler, condition, missing, noise, times, args.batch_size
        )
        if calibration is not None:
            calibrated_states, calibrated_velocity = _path_samples(
                source.sampler, condition, missing, noise, times, args.batch_size,
                calibration=calibration,
            )
        candidate_samples = {
            name: _path_samples(
                source.sampler, condition, missing, noise, times, args.batch_size,
                calibration=adapter, include_endpoint=score_endpoints,
            )
            for name, adapter in rank_candidates.items()
        }
        if score_endpoints:
            truth = source.target_expression[spot, missing].cpu().numpy()
            truth = np.clip((truth + 1.0) / 2.0, 0.0, 1.0)
            baseline_mse += _endpoint_mse(source_samples[2], truth, args.fit_samples)
            for name, samples in candidate_samples.items():
                candidate_mse[name] += _endpoint_mse(samples[2], truth, args.fit_samples)
        for time_index, time in enumerate(times):
            if velocity_slices is not None and time_index not in plot_times:
                continue
            calibrated_source = (
                None if calibration is None else
                (calibrated_states[time_index], calibrated_velocity[time_index])
            )
            scores, path_losses, velocity_slice = _compare_time_and_candidates(
                source_states[time_index], source_velocity[time_index],
                target_states[time_index], target_velocity[time_index],
                args.fit_samples, waves, probe_waves, args.ridge,
                calibrated_source=calibrated_source,
                rank_sources={
                    name: (samples[0][time_index], samples[1][time_index])
                    for name, samples in candidate_samples.items()
                },
                capture_velocity=velocity_slices is not None,
            )
            if velocity_slices is not None and velocity_slice is not None:
                velocity_slices[time] = velocity_slice
            for name, loss in path_losses.items():
                candidate_path_loss[name] += loss
            rows.append({
                "spot_index": int(spot), "spot": source.spot_names[spot], "time": time,
                "target_seed": target_seed,
                **scores,
            })
    endpoint_losses = None
    if score_endpoints:
        endpoint_losses = (
            baseline_mse / len(spots),
            {name: loss / len(spots) for name, loss in candidate_mse.items()},
        )
    return rows, {name: loss / len(rows) for name, loss in candidate_path_loss.items()}, endpoint_losses


def _plot_velocity_fields(
    slices_by_seed: Mapping[int, Mapping[float, VelocitySlice]],
    *,
    fold: int,
    spot: int,
    max_arrows: int,
    destination: Path,
) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    from sklearn.decomposition import PCA

    seeds = sorted(slices_by_seed)
    times = sorted(slices_by_seed[seeds[0]])
    points = np.concatenate([slices_by_seed[seed][time].midpoint for seed in seeds for time in times])
    pca = PCA(n_components=2, svd_solver="full").fit(points)
    projected = pca.transform(points)
    projection = pca.components_.T

    panels = []
    for seed in seeds:
        for time in times:
            item = slices_by_seed[seed][time]
            base = pca.transform(item.midpoint)
            # Spread the arrow tails over the sampled cloud by farthest-point selection.
            index = int(np.argmin(np.sum((base - base.mean(axis=0)) ** 2, axis=1)))
            chosen = []
            nearest = np.full(len(base), np.inf)
            for _ in range(min(max_arrows, len(base))):
                chosen.append(index)
                nearest = np.minimum(nearest, np.sum((base - base[index]) ** 2, axis=1))
                index = int(nearest.argmax())
                if nearest[index] <= 0:
                    break
            panels.append((
                base, base[chosen], item.source[chosen] @ projection, item.target[chosen] @ projection,
            ))

    # Use one conversion from velocity to arrow length across all panels.
    span = max(float(np.ptp(projected, axis=0).max()), 1e-6)
    vectors = np.concatenate([vectors for _, _, source, target in panels for vectors in (source, target)])
    max_speed = max(float(np.linalg.norm(vectors, axis=1).max()), 1e-8)
    length_scale = 0.18 * span / max_speed
    limits = np.stack((projected.min(axis=0) - 0.24 * span, projected.max(axis=0) + 0.24 * span))
    source_color = "#4B5563"
    target_colors = ("#D9E7F4", "#FFC7CE", "#FFF9D9")
    with plt.rc_context({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
    }):
        figure, axes = plt.subplots(
            len(seeds), len(times), squeeze=False, sharex=True, sharey=True,
            figsize=(3.2 * len(times) + 0.8, 3.0 * len(seeds) + 0.95),
        )
        try:
            for row, seed in enumerate(seeds):
                for column, time in enumerate(times):
                    axis = axes[row, column]
                    base, anchors, source, target = panels[row * len(times) + column]
                    axis.scatter(*base.T, s=5, color="#CDD3DA", linewidths=0)
                    for velocity, color, width in (
                        (source, source_color, 0.010), (target, target_colors[row], 0.007),
                    ):
                        axis.quiver(
                            *anchors.T, *(velocity * length_scale).T,
                            angles="xy", scale_units="xy", scale=1, pivot="tail", width=width,
                            color=color, edgecolor="#374151", linewidth=0.45,
                        )
                    axis.scatter(*anchors.T, s=9, color="#374151", linewidths=0, zorder=4)
                    axis.set(xlim=limits[:, 0], ylim=limits[:, 1], aspect="equal")
                    if row == 0:
                        axis.set_title(fr"$t={time:.3g}$")
                    if column == 0:
                        axis.set_ylabel(f"Target {seed}\nPC 2")
                    if row == len(seeds) - 1:
                        axis.set_xlabel("PC 1")
            handles = [Patch(facecolor=source_color, edgecolor="#374151", label="Source flow")]
            handles.extend(
                Patch(facecolor=color, edgecolor="#374151", label=f"Target {seed}")
                for seed, color in zip(seeds, target_colors)
            )
            figure.suptitle(f"Velocity fields at shared probes (fold {fold}, ST spot {spot})")
            figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
            figure.tight_layout(rect=(0, 0.06, 1, 0.95))
            temporary = destination.with_suffix(".pdf.tmp")
            try:
                figure.savefig(temporary, format="pdf")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        finally:
            plt.close(figure)


@torch.no_grad()
def _calibration_rank_scores(
    evaluation: CalibrationRankEvaluation,
    source,
    spots: list[int],
    baseline_path_losses: Mapping[int, float],
    candidate_path_losses: Mapping[int, dict[str, float]],
    endpoint_losses: tuple[float, dict[str, float]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    try:
        from .metric import calibration_rank_correlations, fixed_rbf_mmd_squared
    except ImportError:
        from metric import calibration_rank_correlations, fixed_rbf_mmd_squared

    target_expression = source.target_expression
    target_observed = target_expression[spots][:, source.observed_mask.to(target_expression.device)]
    source_observed = torch.as_tensor(
        evaluation.source_observed, dtype=target_observed.dtype, device=target_observed.device
    ).detach()
    if source_observed.ndim != 2 or source_observed.shape[1] != target_observed.shape[1]:
        raise ValueError("Held-out source observations must match the compact observed gene order.")
    target_samples = target_observed.cpu().numpy()
    baseline_mmd = fixed_rbf_mmd_squared(
        source_observed.cpu().numpy(), target_samples, bandwidth=evaluation.mmd_bandwidth
    )
    baseline_mse, candidate_mse = endpoint_losses
    baseline_path_loss = float(np.mean(list(baseline_path_losses.values())))
    rows = []
    for name, adapter in evaluation.candidates.items():
        inverse_conditions = torch.stack([
            adapter.inverse_condition(condition.clone()) for condition in target_observed
        ])
        round_trip = adapter.forward_condition(inverse_conditions)
        if round_trip.shape != target_observed.shape or not torch.allclose(
            round_trip, target_observed, atol=1e-6
        ):
            raise ValueError(f"Calibration {name!r} must round-trip the evaluated target conditions.")
        transformed = adapter.forward_condition(source_observed.clone())
        if transformed.shape != source_observed.shape:
            raise ValueError(f"Calibration {name!r} must preserve the observed batch shape.")
        candidate_mmd = fixed_rbf_mmd_squared(
            transformed.detach().cpu().numpy(), target_samples, bandwidth=evaluation.mmd_bandwidth
        )
        candidate_path = float(np.mean([losses[name] for losses in candidate_path_losses.values()]))
        rows.append({
            "candidate": name,
            "baseline_mmd_squared": baseline_mmd,
            "candidate_mmd_squared": candidate_mmd,
            "delta_mmd_squared": baseline_mmd - candidate_mmd,
            "baseline_path_loss": baseline_path_loss,
            "candidate_path_loss": candidate_path,
            "delta_path_loss": baseline_path_loss - candidate_path,
            "baseline_mse": baseline_mse,
            "candidate_mse": candidate_mse[name],
            "delta_mse": baseline_mse - candidate_mse[name],
        })
    table = pd.DataFrame(rows)
    return table, {
        **calibration_rank_correlations(table),
        "delta_convention": "baseline_minus_candidate",
        "mmd_bandwidth": evaluation.mmd_bandwidth,
        "heldout_source_observations": len(source_observed),
        "heldout_target_spots": spots,
        "target_seed_count": len(candidate_path_losses),
        "by_target_seed": [
            {
                "target_seed": seed,
                "baseline_path_loss": baseline_path_losses[seed],
                "candidates": {
                    name: {
                        "candidate_path_loss": loss,
                        "delta_path_loss": baseline_path_losses[seed] - loss,
                    }
                    for name, loss in candidate_path_losses[seed].items()
                },
            }
            for seed in sorted(candidate_path_losses)
        ],
        "comparison_measure": "fixed_baseline_midpoint_per_target_seed",
        "endpoint_mse": "mean_evaluation_draws_then_clip_to_unit_interval_on_missing_genes",
    }


def run_comparison(
    args: argparse.Namespace,
    *,
    calibration: CalibrationAdapter | None = None,
    rank_evaluation: CalibrationRankEvaluation | None = None,
) -> None:
    """Evaluate fitted flows and optionally compare supplied calibrations."""
    device = torch.device("cpu") if args.check_only else resolve_device(args.device)
    plot_velocity = getattr(args, "plot_velocity", False)
    rank_candidates = rank_evaluation.candidates if rank_evaluation is not None else {}
    if rank_evaluation is not None and not args.check_only:
        if not rank_candidates or any(not isinstance(name, str) or not name.strip() for name in rank_candidates):
            raise ValueError("Provide at least one calibration candidate with a nonempty name.")
        if not np.isfinite(rank_evaluation.mmd_bandwidth) or rank_evaluation.mmd_bandwidth <= 0:
            raise ValueError("The MMD bandwidth must be positive and finite.")
    rows: list[dict[str, object]] = []
    targets: dict[int, dict[str, object]] = {}
    baseline_path_losses: dict[int, float] = {}
    candidate_path_losses: dict[int, dict[str, float]] = {}
    endpoint_losses = None
    output = None
    velocity_by_seed: dict[int, dict[float, VelocitySlice]] = {}
    plotted_spot = None
    for source, target_sampler, target, test_indices, target_dir, seed in _load_pairs(args, device):
        print(
            f"Flow checkpoints ready for fold {args.fold}, target seed {seed}: "
            f"{len(test_indices)} held-out ST spots."
        )
        if not args.check_only:
            if output is None:
                if getattr(args, "calibration_dir", None) is not None:
                    rank_evaluation = _load_prepared_calibration(args, source, test_indices)
                    rank_candidates = rank_evaluation.candidates
                output = ensure_output_dir(args.output_dir)
            if plot_velocity:
                velocity_by_seed[seed] = {}
                seed_rows, _, _ = _compare_seed(
                    args, source, target_sampler, test_indices, seed, None, {}, False,
                    velocity_slices=velocity_by_seed[seed],
                )
                plotted_spot = int(seed_rows[0]["spot_index"])
                del target_sampler, target
                continue
            seed_rows, seed_candidate_losses, seed_endpoint_losses = _compare_seed(
                args, source, target_sampler, test_indices, seed, calibration,
                rank_candidates, score_endpoints=bool(rank_candidates) and endpoint_losses is None,
            )
            rows.extend(seed_rows)
            if rank_candidates:
                baseline_path_losses[seed] = sum(
                    row["half_squared_velocity_gap"] for row in seed_rows
                ) / len(seed_rows)
                candidate_path_losses[seed] = seed_candidate_losses
            if seed_endpoint_losses is not None:
                endpoint_losses = seed_endpoint_losses
            targets[seed] = {
                "target_seed": seed,
                "target_run_dir": str(target_dir),
                "target_validation": target.get("target_validation_performance"),
            }
        del target_sampler, target
    if args.check_only:
        return
    if plot_velocity:
        destination = output / f"velocity_fold{args.fold:02d}_spot{plotted_spot}.pdf"
        _plot_velocity_fields(
            velocity_by_seed, fold=args.fold, spot=plotted_spot,
            max_arrows=args.velocity_arrows, destination=destination,
        )
        print(f"Velocity plot saved to {destination}.")
        return

    metrics = pd.DataFrame(rows)
    score_columns = metrics.columns[4:].tolist()
    primary_metric = "half_squared_velocity_gap"
    seed_scores = [primary_metric]
    if calibration is not None:
        seed_scores.extend(("calibrated_half_squared_velocity_gap", "half_squared_velocity_gap_reduction"))
    by_seed = metrics.groupby("target_seed")[score_columns].mean()
    by_seed_time = metrics.groupby(["target_seed", "time"])[score_columns].mean()
    by_time = by_seed_time.groupby(level="time")
    if rank_evaluation is not None:
        rank_scores, rank_summary = _calibration_rank_scores(
            rank_evaluation, source, sorted(metrics["spot_index"].unique().tolist()),
            baseline_path_losses, candidate_path_losses, endpoint_losses,
        )
        rank_summary["endpoint_evaluation_draws"] = args.evaluation_samples
        rank_scores.to_csv(output / "calibration_rank_scores.csv", index=False)
        save_json(rank_summary, output / "calibration_rank_summary.json")
    metrics.to_csv(output / "path_comparison.csv", index=False)
    save_json({
        "measure": "fixed_path_minimum_kinetic_velocity_gap",
        "primary_metric": primary_metric,
        "diagnostic_metrics": [
            name for name in score_columns
            if name == "w2_squared_per_gene" or "weak_residual_rms" in name
        ],
        "alignment": "identity",
        "metric": "euclidean_missing_gene_coordinates",
        "comparison_measure": "empirical_wasserstein_midpoint",
        "calibrated_comparison_measure": "fixed_baseline_midpoint" if calibration is not None else None,
        "preprocessing": source.checkpoint["preprocessing"],
        "source_augmentation": source.checkpoint.get("augment_source", True),
        "source_validation": source.checkpoint.get("source_performance"),
        "target_validation": (
            next(iter(targets.values()))["target_validation"] if len(targets) == 1 else None
        ),
        "target_seed_count": len(targets),
        "by_target_seed": [
            {**targets[seed], "mean": scores.to_dict()}
            for seed, scores in by_seed.iterrows()
        ],
        "missing_genes": [gene for gene, keep in zip(source.genes, source.missing_mask) if keep],
        "test_spots": len(test_indices),
        "evaluated_spots": metrics["spot_index"].nunique(),
        "settings": vars(args),
        "mean": by_seed.mean().to_dict(),
        "std": (
            by_seed[seed_scores].std(ddof=1).to_dict()
            if len(targets) > 1 else dict.fromkeys(seed_scores)
        ),
        "by_time": by_time.mean().reset_index().to_dict("records"),
        "std_by_time": (
            by_time[seed_scores].std(ddof=1).reset_index().to_dict("records")
            if len(targets) > 1 else None
        ),
    }, output / "summary.json")
    print(f"Path comparison saved to {output}.")


def main(
    argv: Sequence[str] | None = None,
    *,
    calibration: CalibrationAdapter | None = None,
    rank_evaluation: CalibrationRankEvaluation | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument(
        "--target-run-dir", action="append", required=True,
        help="Target flow run. Repeat the flag to compare multiple target training seeds.",
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--sc-path")
    parser.add_argument("--spatial-path")
    parser.add_argument("--check-only", action="store_true", help="Check prerequisite runs on CPU.")
    parser.add_argument("--output-dir")
    parser.add_argument("--calibration-dir", help="Evaluate frozen candidates from prepare_observed_calibration.py.")
    parser.add_argument(
        "--plot-velocity", action="store_true",
        help="Write a PDF of the source and target velocities at shared identity midpoints.",
    )
    parser.add_argument("--velocity-spot", type=int, help="Sampled ST row index (default: first sampled row).")
    parser.add_argument("--velocity-arrows", type=int, default=8, help="Maximum arrow pairs per panel.")
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
    if args.calibration_dir and (args.check_only or calibration is not None or rank_evaluation is not None):
        parser.error("--calibration-dir requires an evaluation with no Python-supplied adapters.")
    if args.plot_velocity:
        if args.check_only or args.calibration_dir or calibration is not None or rank_evaluation is not None:
            parser.error("--plot-velocity is a separate mode for the identity baseline.")
        if len(args.target_run_dir) > 3:
            parser.error("--plot-velocity accepts up to three target runs.")
        if args.velocity_arrows < 1 or (args.velocity_spot is not None and args.velocity_spot < 0):
            parser.error("Velocity arrow count must be positive and the spot index must be nonnegative.")
        if shutil.which("latex") is None:
            parser.error("--plot-velocity requires LaTeX with amsmath and 'latex' on PATH.")
    if min(args.fold, args.max_spots, args.num_times, args.batch_size, args.ode_steps) < 1:
        parser.error("Fold, spots, times, batch size and ODE steps must be positive.")
    if min(args.fit_samples, args.evaluation_samples) < 2 or args.rff_features < 0:
        parser.error("Fit and evaluation require at least two samples each; RFF feature count must be nonnegative.")
    if not (np.isfinite(args.ridge) and args.ridge > 0) or args.seed < 0:
        parser.error("Ridge must be positive and finite; seed must be nonnegative.")
    run_comparison(args, calibration=calibration, rank_evaluation=rank_evaluation)


if __name__ == "__main__":
    main()
