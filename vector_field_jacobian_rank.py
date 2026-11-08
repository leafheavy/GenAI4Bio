"""Jacobian-rank experiments for mean-only initialization adaptation.

The default experiment trains an offline, mu-conditioned target vector field
from target-domain expression and then evaluates the fixed-state derivative

    partial (u_target(x,t;c,mu) - u_source(x,t;c)) / partial mu.

The earlier source-pathwise derivative remains available as a diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from .initialization_adaptation import (
        MeanOnlyGaussianInitialization,
        _load_cli_tensor,
        _resolve_cli_device,
    )
    from .random_re_masking import (
        _prepare_output_dir,
        build_masked_condition,
        sample_remask_split,
    )
    from .source_flow import (
        build_source_flow_model,
        conditional_flow_matching_loss,
        evaluate_flow_imputation,
        evaluate_flow_matching_loss,
        freeze_source_flow_model,
        load_source_flow_run,
        require_flow_performance,
    )
except ImportError:
    from initialization_adaptation import (
        MeanOnlyGaussianInitialization,
        _load_cli_tensor,
        _resolve_cli_device,
    )
    from random_re_masking import (
        _prepare_output_dir,
        build_masked_condition,
        sample_remask_split,
    )
    from source_flow import (
        build_source_flow_model,
        conditional_flow_matching_loss,
        evaluate_flow_imputation,
        evaluate_flow_matching_loss,
        freeze_source_flow_model,
        load_source_flow_run,
        require_flow_performance,
    )


TARGET_FLOW_METHOD = "mu_conditioned_target_flow"


@dataclass(frozen=True)
class JacobianSpectrum:
    singular_values: torch.Tensor
    numerical_rank: int
    rank_limit: int
    rank_fraction: float
    stable_rank: float
    effective_rank: float
    threshold: float
    is_lower_bound: bool


def _summarize_spectrum(
    singular_values: torch.Tensor,
    *,
    rank_limit: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    is_lower_bound: bool,
) -> JacobianSpectrum:
    values = singular_values.detach()
    largest = float(values[0].item()) if values.numel() else 0.0
    threshold = max(absolute_tolerance, relative_tolerance * largest)
    numerical_rank = int((values > threshold).sum().item())
    squared = values.square()
    stable_rank = (
        float(squared.sum().item() / squared[0].item())
        if values.numel() and squared[0].item() > 0
        else 0.0
    )
    total = values.sum()
    if total.item() > 0:
        probabilities = values / total
        positive = probabilities > 0
        entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
        effective_rank = float(entropy.exp().item())
    else:
        effective_rank = 0.0
    return JacobianSpectrum(
        singular_values=values,
        numerical_rank=numerical_rank,
        rank_limit=rank_limit,
        rank_fraction=(numerical_rank / rank_limit if rank_limit else 0.0),
        stable_rank=stable_rank,
        effective_rank=effective_rank,
        threshold=threshold,
        is_lower_bound=is_lower_bound,
    )


def compute_jacobian_spectrum(
    function: Callable[[torch.Tensor], torch.Tensor],
    mu: torch.Tensor,
    *,
    mode: str = "randomized",
    num_probes: int = 64,
    generator: Optional[torch.Generator] = None,
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 1e-6,
) -> tuple[JacobianSpectrum, torch.Tensor]:
    """Compute an exact Jacobian or a randomized range-rank lower bound."""

    if mu.ndim != 1:
        raise ValueError("mu must be one-dimensional.")
    if mode not in {"exact", "randomized"}:
        raise ValueError("mode must be 'exact' or 'randomized'.")
    if num_probes < 1:
        raise ValueError("num_probes must be positive.")
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("Rank tolerances must be non-negative.")

    output_size = function(mu).numel()
    input_size = mu.numel()
    if mode == "exact":
        jacobian = torch.autograd.functional.jacobian(
            function,
            mu,
            vectorize=True,
            create_graph=False,
            strict=False,
        ).reshape(output_size, input_size)
        singular_values = torch.linalg.svdvals(jacobian)
        return (
            _summarize_spectrum(
                singular_values,
                rank_limit=min(output_size, input_size),
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                is_lower_bound=False,
            ),
            jacobian.detach(),
        )

    probe_count = min(num_probes, input_size)
    generator_device = (
        torch.device(generator.device)
        if generator is not None
        else torch.device("cpu")
    )
    random_matrix = torch.randn(
        (input_size, probe_count),
        device=generator_device,
        dtype=mu.dtype,
        generator=generator,
    ).to(mu.device)
    directions = torch.linalg.qr(random_matrix, mode="reduced").Q
    products = []
    for direction in directions.unbind(1):
        _, product = torch.autograd.functional.jvp(
            function,
            mu,
            direction,
            create_graph=False,
            strict=False,
        )
        products.append(product.reshape(-1))
    projected_jacobian = torch.stack(products, dim=1)
    singular_values = torch.linalg.svdvals(projected_jacobian)
    exact_if_square = probe_count == input_size
    return (
        _summarize_spectrum(
            singular_values,
            rank_limit=min(output_size, probe_count),
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            is_lower_bound=not exact_if_square,
        ),
        projected_jacobian.detach(),
    )


def build_mu_conditioned_condition(
    expression_condition: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    """Concatenate the masked expression condition and one shared global mean."""

    if expression_condition.ndim != 2:
        raise ValueError("expression_condition must have shape [B,G].")
    flat_mean = mean.reshape(-1).to(
        device=expression_condition.device,
        dtype=expression_condition.dtype,
    )
    if flat_mean.numel() != expression_condition.shape[1]:
        raise ValueError("mean dimension must equal the expression gene dimension.")
    expanded_mean = flat_mean.reshape(1, -1).expand(expression_condition.shape[0], -1)
    return torch.cat((expression_condition, expanded_mean), dim=1)


def _cpu_random(
    shape: tuple[int, ...],
    *,
    normal: bool,
    generator: torch.Generator,
    reference: torch.Tensor,
) -> torch.Tensor:
    function = torch.randn if normal else torch.rand
    return function(
        shape,
        generator=generator,
        device="cpu",
        dtype=reference.dtype,
    ).to(reference.device)


def train_mu_conditioned_target_flow(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    mu_center: torch.Tensor,
    mu_train_scale: float,
    learning_rate: float,
    epochs: int,
    device: torch.device,
    generator: torch.Generator,
    show_progress: bool,
    validation_dataloader: Optional[torch.utils.data.DataLoader] = None,
    validation_seed: int = 0,
) -> list[dict[str, float]]:
    """Train the target field family with one dataset-level mean per batch."""

    if epochs < 1 or learning_rate <= 0 or mu_train_scale <= 0:
        raise ValueError("epochs, learning_rate, and mu_train_scale must be positive.")
    center = mu_center.detach().reshape(1, -1).to(device)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=0.0)
    iterator = (
        tqdm(range(epochs), ncols=100, desc="train target mu-flow")
        if show_progress
        else range(epochs)
    )
    history = []
    for epoch in iterator:
        total_loss = 0.0
        batch_count = 0
        for target, expression_condition in dataloader:
            target = target.float().to(device)
            expression_condition = expression_condition.float().to(device)
            sampled_mean = center + mu_train_scale * _cpu_random(
                tuple(center.shape),
                normal=True,
                generator=generator,
                reference=center,
            )
            epsilon = _cpu_random(
                tuple(target.shape),
                normal=True,
                generator=generator,
                reference=target,
            )
            times = _cpu_random(
                (target.shape[0],),
                normal=False,
                generator=generator,
                reference=target,
            )
            initial_state = sampled_mean + epsilon
            full_condition = build_mu_conditioned_condition(
                expression_condition, sampled_mean
            )
            optimizer.zero_grad(set_to_none=True)
            loss, _ = conditional_flow_matching_loss(
                model,
                target,
                full_condition,
                source_noise=initial_state,
                times=times,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().item())
            batch_count += 1
        if batch_count == 0:
            raise RuntimeError("The target-flow DataLoader produced no batches.")
        mean_loss = total_loss / batch_count
        record = {"epoch": float(epoch + 1), "flow_matching_loss": mean_loss}
        if validation_dataloader is not None:
            record["validation_flow_matching_loss"] = evaluate_flow_matching_loss(
                model,
                validation_dataloader,
                device=device,
                seed=validation_seed,
                initial_mean=mu_center,
            )
        history.append(record)
        if show_progress:
            validation_text = (
                ""
                if validation_dataloader is None
                else f", val:{record['validation_flow_matching_loss']:.6f}"
            )
            iterator.set_postfix_str(
                f"target cfm loss:{mean_loss:.6f}{validation_text}"
            )
    return history


def _split_target_spots(
    num_spots: int,
    train_fraction: float,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_spots < 2:
        raise ValueError("At least two target spots are required for train/evaluation split.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("target_train_fraction must be strictly between 0 and 1.")
    train_count = min(num_spots - 1, max(1, int(round(num_spots * train_fraction))))
    permutation = torch.randperm(num_spots, generator=generator)
    return permutation[:train_count], permutation[train_count:]


def _target_model_config(source_checkpoint: dict[str, object]) -> dict[str, object]:
    config = dict(source_checkpoint["model_config"])
    dimension = int(config["input_size"])
    config["condition_size"] = 2 * dimension
    return config


def _condition_mask_for_new_run(
    observed_mask: torch.Tensor,
    *,
    condition_mode: str,
    remask_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if condition_mode == "oracle":
        return observed_mask
    split = sample_remask_split(
        observed_mask,
        remask_ratio,
        generator=generator,
    )
    return split.condition_mask


def _load_target_flow_checkpoint(
    checkpoint_path: str,
    *,
    loaded_source,
    condition_mode: str,
    condition_fill_value: float,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object], torch.Tensor, torch.Tensor, torch.Tensor]:
    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Target-flow checkpoint not found: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != TARGET_FLOW_METHOD:
        raise ValueError(
            f"--target-flow-checkpoint expects a {TARGET_FLOW_METHOD} checkpoint "
            f"(target_mu_conditioned_flow.pt); {source} has method "
            f"{checkpoint.get('method')!r}. The source-flow checkpoint is loaded "
            "automatically from --source-flow-run-dir. Omit "
            "--target-flow-checkpoint to train the target oracle flow."
        )
    if int(checkpoint["fold"]) != int(loaded_source.checkpoint["fold"]):
        raise ValueError("Target-flow checkpoint fold differs from source-flow fold.")
    if checkpoint["genes"] != list(loaded_source.genes):
        raise ValueError("Target-flow checkpoint gene order differs from source run.")
    if checkpoint.get("preprocessing", "legacy_full_gene_v1") != loaded_source.checkpoint.get(
        "preprocessing", "legacy_full_gene_v1"
    ):
        raise ValueError("Target-flow checkpoint preprocessing differs from source run.")
    if checkpoint["condition_mode"] != condition_mode:
        raise ValueError("Target-flow checkpoint condition mode differs from the CLI.")
    if float(checkpoint["condition_fill_value"]) != float(condition_fill_value):
        raise ValueError("Target-flow checkpoint condition fill value differs from the CLI.")
    require_flow_performance(
        checkpoint,
        label=f"Target-flow checkpoint {source}",
        performance_key="performance",
        expected_evaluation_domain="held_out_target_spatial",
    )
    condition_mask = torch.as_tensor(checkpoint["condition_mask"], dtype=torch.bool)
    if condition_mask.numel() != len(loaded_source.genes):
        raise ValueError("Target-flow checkpoint condition mask has the wrong size.")
    model = build_source_flow_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    freeze_source_flow_model(model)
    return (
        model,
        checkpoint,
        condition_mask,
        torch.as_tensor(checkpoint["train_indices"], dtype=torch.long),
        torch.as_tensor(checkpoint["evaluation_indices"], dtype=torch.long),
    )


def _train_target_flow(
    *,
    loaded_source,
    condition_mask: torch.Tensor,
    train_indices: torch.Tensor,
    evaluation_indices: torch.Tensor,
    mu_center: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> tuple[nn.Module, dict[str, object], Path]:
    try:
        from .Dataloader import get_data_loader
        from .utils import set_seed
    except ImportError:
        from Dataloader import get_data_loader
        from utils import set_seed

    full_condition = build_masked_condition(
        loaded_source.target_expression,
        condition_mask,
        fill_value=args.condition_fill_value,
    )
    train_expression = loaded_source.target_expression.index_select(
        0, train_indices.to(device)
    )
    train_condition = full_condition.index_select(0, train_indices.to(device))
    evaluation_expression = loaded_source.target_expression.index_select(
        0, evaluation_indices.to(device)
    )
    evaluation_condition = full_condition.index_select(
        0, evaluation_indices.to(device)
    )
    dataloader = get_data_loader(
        train_expression.detach().cpu().numpy(),
        train_condition.detach().cpu().numpy(),
        batch_size=args.target_batch_size,
        is_shuffle=True,
        seed=args.seed,
        num_workers=args.target_num_workers,
    )
    validation_dataloader = get_data_loader(
        evaluation_expression.detach().cpu().numpy(),
        evaluation_condition.detach().cpu().numpy(),
        batch_size=args.target_validation_batch_size,
        is_shuffle=False,
        seed=args.seed,
        num_workers=args.target_num_workers,
    )
    model_config = _target_model_config(loaded_source.checkpoint)
    set_seed(args.seed, deterministic=args.deterministic)
    model = build_source_flow_model(model_config)
    training_generator = torch.Generator(device="cpu").manual_seed(args.seed + 11)
    history = train_mu_conditioned_target_flow(
        model,
        dataloader,
        mu_center=mu_center,
        mu_train_scale=args.mu_train_scale,
        learning_rate=args.target_learning_rate,
        epochs=args.target_flow_epochs,
        device=device,
        generator=training_generator,
        show_progress=not args.no_progress,
        validation_dataloader=validation_dataloader,
        validation_seed=args.seed + 21,
    )
    freeze_source_flow_model(model)
    target_gene_indices = torch.nonzero(~condition_mask, as_tuple=True)[0]
    performance = evaluate_flow_imputation(
        model,
        evaluation_expression,
        evaluation_condition,
        target_gene_indices=target_gene_indices,
        baseline_mean=train_expression.mean(dim=0),
        genes=list(loaded_source.genes),
        spot_names=[
            loaded_source.spot_names[index]
            for index in evaluation_indices.tolist()
        ],
        validation_cfm_loss=history[-1]["validation_flow_matching_loss"],
        device=device,
        batch_size=args.target_validation_batch_size,
        num_noise_samples=args.target_validation_num_noise_samples,
        seed=args.seed + 22,
        ode_method=args.ode_method,
        ode_steps=args.ode_steps,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        max_relative_mse=args.target_max_validation_relative_mse,
        min_median_spcc=args.target_min_validation_spcc,
        output_dir=output_dir / "target_flow_performance",
        initial_mean=mu_center,
        metadata={
            "evaluation_domain": "held_out_target_spatial",
            "baseline": "target_training_gene_mean",
            "condition_mode": args.condition_mode,
        },
    ).metrics
    checkpoint = {
        "method": TARGET_FLOW_METHOD,
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "fold": int(loaded_source.checkpoint["fold"]),
        "genes": list(loaded_source.genes),
        "preprocessing": loaded_source.checkpoint.get(
            "preprocessing", "legacy_full_gene_v1"
        ),
        "condition_mode": args.condition_mode,
        "condition_mask": condition_mask.tolist(),
        "condition_fill_value": args.condition_fill_value,
        "remask_ratio": args.remask_ratio if args.condition_mode == "remask" else None,
        "train_indices": train_indices.tolist(),
        "evaluation_indices": evaluation_indices.tolist(),
        "mu_center": mu_center.detach().cpu(),
        "mu_train_scale": args.mu_train_scale,
        "target_flow_epochs": args.target_flow_epochs,
        "target_batch_size": args.target_batch_size,
        "target_learning_rate": args.target_learning_rate,
        "target_train_fraction": args.target_train_fraction,
        "performance": performance,
        "seed": args.seed,
    }
    checkpoint_path = output_dir / "target_mu_conditioned_flow.pt"
    torch.save(checkpoint, checkpoint_path)
    with (output_dir / "target_flow_train_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "flow_matching_loss",
                "validation_flow_matching_loss",
            ),
        )
        writer.writeheader()
        writer.writerows(history)
    if not performance["passed"]:
        raise RuntimeError(
            "Target oracle-flow performance gate failed; Jacobian rank was not "
            "computed. See target_flow_performance/flow_performance.json."
        )
    return model, checkpoint, checkpoint_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train/load a mu-conditioned target flow and estimate fixed-state "
            "target-source vector-field Jacobian rank."
        )
    )
    parser.add_argument("--source-flow-run-dir", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--sc-path", default=None)
    parser.add_argument("--spatial-path", default=None)
    parser.add_argument(
        "--jacobian-definition",
        choices=("target-fixed-state", "source-pathwise"),
        default="target-fixed-state",
    )
    parser.add_argument("--target-flow-checkpoint", default=None)
    parser.add_argument("--target-flow-epochs", type=int, default=900)
    parser.add_argument("--target-batch-size", type=int, default=128)
    parser.add_argument("--target-learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-train-fraction", type=float, default=0.8)
    parser.add_argument("--target-validation-batch-size", type=int, default=32)
    parser.add_argument("--target-validation-num-noise-samples", type=int, default=4)
    parser.add_argument(
        "--target-max-validation-relative-mse", type=float, default=1.0
    )
    parser.add_argument("--target-min-validation-spcc", type=float, default=0.0)
    parser.add_argument("--mu-train-scale", type=float, default=0.1)
    parser.add_argument("--target-num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--condition-mode", choices=("oracle", "remask"), default="oracle")
    parser.add_argument("--remask-ratio", type=float, default=0.2)
    parser.add_argument(
        "--times", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--num-spots", type=int, default=1)
    parser.add_argument(
        "--jacobian-mode", choices=("exact", "randomized"), default="randomized"
    )
    parser.add_argument("--num-probes", type=int, default=64)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--mu", default=None)
    parser.add_argument("--noise", default=None)
    parser.add_argument("--condition-fill-value", type=float, default=-1.0)
    parser.add_argument("--ode-method", default="rk4")
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--ode-rtol", type=float, default=1e-5)
    parser.add_argument("--ode-atol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1202)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-existing", action="store_true")
    return parser


def _load_mean(
    path: Optional[str],
    *,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if path is None:
        return torch.zeros(dimension, device=device, dtype=dtype)
    value = _load_cli_tensor(path, key="mu", device=device, dtype=dtype).reshape(-1)
    if value.numel() != dimension:
        raise ValueError(f"mu must contain {dimension} values.")
    return value


def _load_evaluation_noise(
    path: Optional[str],
    *,
    num_spots: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    if path is None:
        adapter = MeanOnlyGaussianInitialization(dimension, device=device, dtype=dtype)
        return adapter.sample_noise(num_spots, generator=generator)
    noise = _load_cli_tensor(path, key="noise", device=device, dtype=dtype)
    if tuple(noise.shape) != (num_spots, dimension):
        raise ValueError(f"noise must have shape {(num_spots, dimension)}.")
    return noise


def _spectrum_record(
    time: float,
    spectrum: JacobianSpectrum,
    *,
    mismatch_norm: Optional[float],
) -> dict[str, object]:
    return {
        "time": float(time),
        "numerical_rank": spectrum.numerical_rank,
        "rank_limit": spectrum.rank_limit,
        "rank_fraction": spectrum.rank_fraction,
        "stable_rank": spectrum.stable_rank,
        "effective_rank": spectrum.effective_rank,
        "threshold": spectrum.threshold,
        "rank_is_lower_bound": spectrum.is_lower_bound,
        "largest_singular_value": (
            float(spectrum.singular_values[0].item())
            if spectrum.singular_values.numel()
            else 0.0
        ),
        "target_source_mismatch_norm": mismatch_norm,
    }


def run_experiment(
    args: argparse.Namespace, *, print_summary: bool = True
) -> dict[str, object]:
    if args.num_spots < 1:
        raise ValueError("num_spots must be positive.")
    if args.target_validation_batch_size < 1:
        raise ValueError("target_validation_batch_size must be positive.")
    if args.target_validation_num_noise_samples < 1:
        raise ValueError("target_validation_num_noise_samples must be positive.")
    if args.target_max_validation_relative_mse <= 0:
        raise ValueError("target_max_validation_relative_mse must be positive.")
    if any(not 0.0 <= value <= 1.0 for value in args.times):
        raise ValueError("All --times values must be in [0,1].")
    output_dir = _prepare_output_dir(
        args.output_dir, allow_existing=args.allow_existing
    )
    device = _resolve_cli_device(args.device)
    loaded = load_source_flow_run(
        args.source_flow_run_dir,
        fold=args.fold,
        device=device,
        sc_path=args.sc_path,
        spatial_path=args.spatial_path,
        ode_method=args.ode_method,
        ode_steps=args.ode_steps,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        require_row_separable=getattr(args, "require_row_separable", False),
        require_observed_only_preprocessing=getattr(
            args, "require_observed_only_preprocessing", False
        ),
    )
    dimension = int(loaded.checkpoint["model_config"]["input_size"])
    mu_center = _load_mean(
        args.mu,
        dimension=dimension,
        device=device,
        dtype=loaded.target_expression.dtype,
    )
    mask_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    split_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    selection_generator = torch.Generator(device="cpu").manual_seed(args.seed + 2)
    noise_generator = torch.Generator(device="cpu").manual_seed(args.seed + 3)
    target_model = None
    target_checkpoint = None
    target_checkpoint_path = None

    if args.jacobian_definition == "target-fixed-state":
        if args.target_flow_checkpoint is not None:
            (
                target_model,
                target_checkpoint,
                condition_mask,
                train_indices,
                evaluation_indices,
            ) = _load_target_flow_checkpoint(
                args.target_flow_checkpoint,
                loaded_source=loaded,
                condition_mode=args.condition_mode,
                condition_fill_value=args.condition_fill_value,
                device=device,
            )
            target_checkpoint_path = Path(args.target_flow_checkpoint).resolve()
            checkpoint_center = torch.as_tensor(
                target_checkpoint["mu_center"],
                device=device,
                dtype=loaded.target_expression.dtype,
            ).reshape(-1)
            if checkpoint_center.numel() != dimension:
                raise ValueError("Target-flow checkpoint mu center has the wrong size.")
            if args.mu is None:
                mu_center = checkpoint_center
        else:
            condition_mask = _condition_mask_for_new_run(
                loaded.observed_mask,
                condition_mode=args.condition_mode,
                remask_ratio=args.remask_ratio,
                generator=mask_generator,
            )
            train_indices, evaluation_indices = _split_target_spots(
                loaded.target_expression.shape[0],
                args.target_train_fraction,
                generator=split_generator,
            )
            target_model, target_checkpoint, target_checkpoint_path = _train_target_flow(
                loaded_source=loaded,
                condition_mask=condition_mask,
                train_indices=train_indices,
                evaluation_indices=evaluation_indices,
                mu_center=mu_center,
                args=args,
                device=device,
                output_dir=output_dir,
            )
        candidate_indices = evaluation_indices
    else:
        if args.target_flow_checkpoint is not None:
            raise ValueError("--target-flow-checkpoint only applies to target-fixed-state.")
        condition_mask = _condition_mask_for_new_run(
            loaded.observed_mask,
            condition_mode=args.condition_mode,
            remask_ratio=args.remask_ratio,
            generator=mask_generator,
        )
        train_indices = torch.empty(0, dtype=torch.long)
        evaluation_indices = torch.arange(loaded.target_expression.shape[0])
        candidate_indices = evaluation_indices

    if args.num_spots > candidate_indices.numel():
        raise ValueError(
            "num_spots exceeds the available held-out target spots; lower "
            "--num-spots or --target-train-fraction."
        )
    selection_order = torch.randperm(
        candidate_indices.numel(), generator=selection_generator
    )
    selected_indices = candidate_indices.index_select(
        0, selection_order[: args.num_spots]
    )
    expression = loaded.target_expression.index_select(0, selected_indices.to(device))
    condition = build_masked_condition(
        expression,
        condition_mask,
        fill_value=args.condition_fill_value,
    )
    noise = _load_evaluation_noise(
        args.noise,
        num_spots=args.num_spots,
        dimension=dimension,
        device=device,
        dtype=expression.dtype,
        generator=noise_generator,
    )
    mu = mu_center.detach().clone().requires_grad_(True)
    rank_generator = torch.Generator(device="cpu").manual_seed(args.seed + 4)
    records = []
    spectra = {}
    jacobian_representations = {}
    anchor_states = {}
    mismatches = {}

    for time in args.times:
        key = f"t_{float(time):.6f}"
        if args.jacobian_definition == "target-fixed-state":
            initial_state = mu_center.reshape(1, -1) + noise
            anchor_state = (
                (1.0 - float(time)) * initial_state + float(time) * expression
            ).detach()
            batch_time = torch.full(
                (args.num_spots,),
                float(time),
                device=device,
                dtype=expression.dtype,
            )
            with torch.no_grad():
                source_velocity = loaded.source_model(
                    anchor_state, batch_time, condition
                ).detach()

            def velocity_from_mean(mean: torch.Tensor) -> torch.Tensor:
                target_condition = build_mu_conditioned_condition(condition, mean)
                target_velocity = target_model(
                    anchor_state, batch_time, target_condition
                )
                return (target_velocity - source_velocity).reshape(-1)

            mismatch = velocity_from_mean(mu).detach()
            mismatch_norm = float(torch.linalg.vector_norm(mismatch).item())
            anchor_states[key] = anchor_state.cpu()
            mismatches[key] = mismatch.cpu()
        else:

            def velocity_from_mean(mean: torch.Tensor) -> torch.Tensor:
                initial_state = mean.reshape(1, -1) + noise
                velocity = loaded.sampler.velocity_at(
                    initial_state,
                    condition,
                    float(time),
                )
                return velocity.reshape(-1)

            mismatch_norm = None

        spectrum, representation = compute_jacobian_spectrum(
            velocity_from_mean,
            mu,
            mode=args.jacobian_mode,
            num_probes=args.num_probes,
            generator=rank_generator,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
        )
        spectra[key] = spectrum.singular_values.cpu()
        jacobian_representations[key] = representation.cpu()
        records.append(
            _spectrum_record(
                float(time), spectrum, mismatch_norm=mismatch_norm
            )
        )

    fixed_state = args.jacobian_definition == "target-fixed-state"
    checkpoint_center_distance = None
    if target_checkpoint is not None:
        checkpoint_center = torch.as_tensor(
            target_checkpoint["mu_center"],
            device=device,
            dtype=mu.dtype,
        ).reshape(-1)
        checkpoint_center_distance = float(
            torch.linalg.vector_norm(mu.detach() - checkpoint_center).item()
        )
    summary = {
        "source_flow_run_dir": str(Path(args.source_flow_run_dir).resolve()),
        "target_flow_checkpoint": (
            str(target_checkpoint_path) if target_checkpoint_path is not None else None
        ),
        "fold": args.fold,
        "condition_mode": args.condition_mode,
        "remask_ratio": args.remask_ratio if args.condition_mode == "remask" else None,
        "jacobian_definition": (
            "partial (u_target(x,t;c,mu)-u_source(x,t;c)) / partial mu "
            "at fixed x,t,c"
            if fixed_state
            else "d u_source(Phi_t(mu+epsilon;c),t,c) / d mu"
        ),
        "jacobian_mode": args.jacobian_mode,
        "num_probes": args.num_probes,
        "num_spots": args.num_spots,
        "spot_indices": selected_indices.tolist(),
        "target_train_indices": train_indices.tolist(),
        "target_evaluation_indices": evaluation_indices.tolist(),
        "mu_train_scale": (
            target_checkpoint.get("mu_train_scale")
            if target_checkpoint is not None
            else None
        ),
        "evaluation_mu_distance_from_training_center": checkpoint_center_distance,
        "source_flow_performance": loaded.checkpoint.get("source_performance"),
        "source_flow_target_transfer_performance": loaded.checkpoint.get(
            "target_transfer_performance"
        ),
        "target_flow_performance": (
            target_checkpoint.get("performance")
            if target_checkpoint is not None
            else None
        ),
        "relative_tolerance": args.relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "seed": args.seed,
        "results": records,
        "theory_note": (
            "The default fixed-state Jacobian is an offline target-domain "
            "approximation of the theoretical vector-field-mismatch derivative. "
            "A posterior-covariance rank equality additionally relies on the "
            "theoretical path assumptions and target-flow estimation quality."
            if fixed_state
            else "The source-pathwise mode is an operational capacity diagnostic."
        ),
    }
    with (output_dir / "jacobian_rank_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_dir / "jacobian_spectrum.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "singular_value_index", "singular_value"])
        for key, values in spectra.items():
            spectrum_time = float(key.removeprefix("t_"))
            for index, value in enumerate(values.tolist(), start=1):
                writer.writerow([spectrum_time, index, value])
    torch.save(
        {
            "mu": mu.detach().cpu(),
            "noise": noise.detach().cpu(),
            "condition_mask": condition_mask,
            "observed_mask": loaded.observed_mask,
            "missing_mask": loaded.missing_mask,
            "genes": loaded.genes,
            "selected_spot_indices": selected_indices,
            "singular_values": spectra,
            "jacobian_representations": jacobian_representations,
            "anchor_states": anchor_states,
            "target_source_mismatches": mismatches,
        },
        output_dir / "jacobian_rank_tensors.pt",
    )
    if print_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_experiment(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()


__all__ = [
    "JacobianSpectrum",
    "build_mu_conditioned_condition",
    "compute_jacobian_spectrum",
    "run_experiment",
    "train_mu_conditioned_target_flow",
]
