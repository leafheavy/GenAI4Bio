"""Offline rank and repairability check for one dataset/fold.

Reuses the target oracle-flow and fixed-state Jacobian experiment in
vector_field_jacobian_rank.py. Target missing-gene truth is used only here for
mechanism analysis, never as an unsupervised adaptation signal.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import torch

try:
    from .vector_field_jacobian_rank import (
        _build_parser,
        _summarize_spectrum,
        run_experiment,
    )
except ImportError:
    from vector_field_jacobian_rank import (
        _build_parser,
        _summarize_spectrum,
        run_experiment,
    )


def _linearized_repair(
    jacobian: torch.Tensor,
    mismatch: torch.Tensor,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    """Solve min_h ||mismatch + J h|| using retained singular directions."""

    matrix = jacobian.detach().to(dtype=torch.float64)
    error = mismatch.detach().reshape(-1).to(dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != error.numel():
        raise ValueError("Jacobian rows must match the vector-field mismatch.")
    if not torch.isfinite(matrix).all() or not torch.isfinite(error).all():
        raise ValueError("Jacobian and mismatch must contain finite values.")
    left, singular_values, right_transposed = torch.linalg.svd(
        matrix, full_matrices=False
    )
    spectrum = _summarize_spectrum(
        singular_values,
        rank_limit=min(matrix.shape),
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        is_lower_bound=False,
    )
    rank = spectrum.numerical_rank
    step = torch.zeros(matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
    if rank:
        coordinates = left[:, :rank].T @ error
        step = -(right_transposed[:rank].T @ (coordinates / singular_values[:rank]))
    residual = error + matrix @ step
    mismatch_norm = float(torch.linalg.vector_norm(error))
    residual_norm = float(torch.linalg.vector_norm(residual))
    residual_ratio = residual_norm / mismatch_norm if mismatch_norm else None
    record = {
        "numerical_rank": rank,
        "rank_limit": spectrum.rank_limit,
        "rank_fraction": spectrum.rank_fraction,
        "stable_rank": spectrum.stable_rank,
        "effective_rank": spectrum.effective_rank,
        "rank_threshold": spectrum.threshold,
        "largest_singular_value": float(singular_values[0]) if singular_values.numel() else 0.0,
        "smallest_retained_singular_value": (
            float(singular_values[rank - 1]) if rank else None
        ),
        "retained_condition_number": (
            float(singular_values[0] / singular_values[rank - 1]) if rank else None
        ),
        "mismatch_norm": mismatch_norm,
        "linearized_residual_norm": residual_norm,
        "linearized_residual_ratio": residual_ratio,
        "first_order_correctable_energy_fraction": (
            max(0.0, 1.0 - residual_ratio**2) if residual_ratio is not None else None
        ),
        "minimum_norm_step_norm": float(torch.linalg.vector_norm(step)),
    }
    return record, step.cpu(), singular_values.cpu()


def _analyze_time(
    jacobian: torch.Tensor,
    mismatch: torch.Tensor,
    *,
    missing_mask: torch.Tensor,
    spot_indices: Sequence[int],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor, torch.Tensor]:
    num_spots = len(spot_indices)
    dimension = missing_mask.numel()
    if tuple(jacobian.shape) != (num_spots * dimension, dimension):
        raise ValueError("Expected an exact [spots * genes, genes] Jacobian.")
    if mismatch.numel() != num_spots * dimension:
        raise ValueError("Target-source mismatch has the wrong size.")
    missing = missing_mask.bool()
    observed = ~missing
    block = jacobian.reshape(num_spots, dimension, dimension).double()
    error = mismatch.reshape(num_spots, dimension).double()
    missing_block = block[:, missing][:, :, missing]
    observed_block = block[:, observed][:, :, missing]
    missing_error = error[:, missing]

    shared, shared_step, singular_values = _linearized_repair(
        missing_block.reshape(-1, int(missing.sum())),
        missing_error.reshape(-1),
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    per_spot = []
    per_spot_steps = []
    independent_residual_squared = 0.0
    for index, (local_jacobian, local_error) in enumerate(
        zip(missing_block, missing_error)
    ):
        local, step, _ = _linearized_repair(
            local_jacobian,
            local_error,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        local["spot_index"] = int(spot_indices[index])
        per_spot.append(local)
        per_spot_steps.append(step)
        independent_residual_squared += local["linearized_residual_norm"] ** 2

    independent_residual_norm = independent_residual_squared**0.5
    mismatch_norm = shared["mismatch_norm"]
    missing_response_norm = float(torch.linalg.matrix_norm(missing_block.reshape(-1, int(missing.sum()))))
    observed_response_norm = float(torch.linalg.matrix_norm(observed_block.reshape(-1, int(missing.sum()))))
    independent_ratio = (
        independent_residual_norm / mismatch_norm if mismatch_norm else None
    )
    result = {
        "shared_mu": shared,
        "per_spot_mu": {
            "linearized_residual_norm": independent_residual_norm,
            "linearized_residual_ratio": independent_ratio,
            "numerical_rank_min": min(item["numerical_rank"] for item in per_spot),
            "numerical_rank_max": max(item["numerical_rank"] for item in per_spot),
            "spots": per_spot,
        },
        "shared_minus_per_spot_residual_ratio": (
            shared["linearized_residual_ratio"] - independent_ratio
            if independent_ratio is not None
            else None
        ),
        "observed_response_over_missing_response": (
            observed_response_norm / missing_response_norm
            if missing_response_norm
            else None
        ),
    }
    return result, shared_step, torch.stack(per_spot_steps), singular_values


def _parser() -> argparse.ArgumentParser:
    parser = _build_parser()
    parser.description = (
        "Offline PDF rank analysis on the missing-gene subspace for one dataset/fold."
    )
    parser.add_argument("--dataset", required=True, help="dataset2 or 2, for example.")
    next(action for action in parser._actions if action.dest == "fold").required = True
    parser.set_defaults(
        fold=None,
        jacobian_mode="exact",
        jacobian_definition="target-fixed-state",
        num_spots=4,
        times=[0.25, 0.5, 0.75],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    dataset = str(args.dataset).strip().lower()
    dataset = dataset if dataset.startswith("dataset") else f"dataset{dataset}"
    if args.fold < 1:
        parser.error("--fold must identify one gene fold, starting at 1.")
    if args.jacobian_definition != "target-fixed-state" or args.jacobian_mode != "exact":
        parser.error("Rank detection requires --jacobian-definition target-fixed-state --jacobian-mode exact.")
    if args.condition_mode != "oracle":
        parser.error("Rank detection uses the original observed-gene condition: --condition-mode oracle.")
    if args.num_spots < 1:
        parser.error("--num-spots must be positive.")
    if any(not 0.0 < time < 1.0 for time in args.times) or len(set(args.times)) != len(args.times):
        parser.error("--times must contain distinct interior values strictly between 0 and 1.")
    source_dir = Path(args.source_flow_run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir == source_dir or source_dir in output_dir.parents:
        parser.error("--output-dir must be outside the source-flow run.")
    with (source_dir / "experiment.json").open(encoding="utf-8") as handle:
        experiment = json.load(handle)
    with (source_dir / "data_info.json").open(encoding="utf-8") as handle:
        data_info = json.load(handle)
    if experiment.get("method") != "source_flow" or data_info.get("dataset") != dataset:
        parser.error("--dataset does not match the source-flow run metadata.")
    if experiment.get("preprocessing") != "gene_local_log1p_v1":
        parser.error("This diagnostic requires a retrained gene_local_log1p_v1 source flow.")
    args.require_row_separable = True
    args.require_observed_only_preprocessing = True

    base = run_experiment(args, print_summary=False)
    saved = torch.load(
        output_dir / "jacobian_rank_tensors.pt", map_location="cpu", weights_only=True
    )
    missing_mask = saved["missing_mask"].bool()
    selected = [int(value) for value in saved["selected_spot_indices"].tolist()]
    records = []
    steps = {}
    spectra = {}
    for original in base["results"]:
        time = float(original["time"])
        key = f"t_{time:.6f}"
        result, shared_step, per_spot_steps, singular_values = _analyze_time(
            saved["jacobian_representations"][key],
            saved["target_source_mismatches"][key],
            missing_mask=missing_mask,
            spot_indices=selected,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
        )
        records.append({"time": time, **result})
        steps[key] = {"shared": shared_step, "per_spot": per_spot_steps}
        spectra[key] = singular_values

    summary = {
        "experiment": "offline_missing_gene_rank_detection_v1",
        "dataset": dataset,
        "fold": args.fold,
        "source_flow_run_dir": str(source_dir),
        "target_flow_checkpoint": base["target_flow_checkpoint"],
        "target_flow_performance": base["target_flow_performance"],
        "num_spots": len(selected),
        "spot_indices": selected,
        "num_missing_genes": int(missing_mask.sum()),
        "missing_genes": [
            gene for gene, is_missing in zip(saved["genes"], missing_mask.tolist())
            if is_missing
        ],
        "relative_tolerance": args.relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "results": records,
        "interpretation": (
            "At fixed x,t,c, project -(u_target-u_source) in missing genes onto "
            "the Jacobian columns for missing-gene mu. Small singular values "
            "or a large per-spot residual suggest local capacity limits; an "
            "additional shared-mu residual suggests one mu cannot satisfy "
            "all selected spots. This uses target full data offline and depends "
            "on target oracle-flow quality; it does not prove condition "
            "non-identifiability or final unsupervised performance."
        ),
    }
    (output_dir / "rank_detection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "rank_detection_spectrum.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "index", "missing_gene_jacobian_singular_value"])
        for record in records:
            values = spectra[f"t_{record['time']:.6f}"]
            for index, value in enumerate(values.tolist(), start=1):
                writer.writerow([record["time"], index, value])
    torch.save(
        {"missing_mask": missing_mask, "selected_spot_indices": saved["selected_spot_indices"],
         "minimum_norm_steps": steps},
        output_dir / "rank_detection_tensors.pt",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
