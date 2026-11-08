"""Conditional source flow model and differentiable ODE sampling utilities.

New source-flow runs use a row-separable backbone whose self-attention is over
genes within each spot.  The original batch-attention stDiff backbone remains
available for loading explicitly tagged or legacy checkpoints.  Both variants
use the same straight conditional flow-matching path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


GENE_ATTENTION_ARCHITECTURE = "gene_attention_row_separable_v1"
LEGACY_BATCH_ATTENTION_ARCHITECTURE = "batch_attention_legacy_v1"
_ARCHITECTURE_ALIASES = {
    GENE_ATTENTION_ARCHITECTURE: GENE_ATTENTION_ARCHITECTURE,
    "gene_attention": GENE_ATTENTION_ARCHITECTURE,
    "row_separable_gene_attention": GENE_ATTENTION_ARCHITECTURE,
    LEGACY_BATCH_ATTENTION_ARCHITECTURE: LEGACY_BATCH_ATTENTION_ARCHITECTURE,
    "batch_attention": LEGACY_BATCH_ATTENTION_ARCHITECTURE,
    "model_original": LEGACY_BATCH_ATTENTION_ARCHITECTURE,
}


def source_flow_architecture(model_config: Mapping[str, Any]) -> str:
    """Resolve a canonical architecture name.

    Configurations written before architecture tagging are treated as legacy
    batch-attention configurations.  New training always writes an explicit
    row-separable architecture tag.
    """

    if not isinstance(model_config, Mapping):
        raise TypeError("model_config must be a mapping.")
    raw = model_config.get("architecture")
    if raw is None:
        return LEGACY_BATCH_ATTENTION_ARCHITECTURE
    key = str(raw).strip().lower()
    try:
        return _ARCHITECTURE_ALIASES[key]
    except KeyError as error:
        raise ValueError(f"Unsupported source-flow architecture: {raw!r}.") from error


def _architecture_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> Optional[str]:
    keys = tuple(str(key) for key in state_dict)
    if any(
        "backbone.gene_embedding." in key or "backbone.value_embedding." in key
        for key in keys
    ):
        return GENE_ATTENTION_ARCHITECTURE
    if any("backbone.in_layer." in key for key in keys):
        return LEGACY_BATCH_ATTENTION_ARCHITECTURE
    return None


def source_flow_model_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a canonical model config and audit its architecture declaration."""

    raw_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Checkpoint is missing a valid model_config mapping.")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint is missing a valid model_state_dict mapping.")
    config = dict(raw_config)
    inferred = _architecture_from_state_dict(state_dict)
    if config.get("architecture") is None:
        config["architecture"] = inferred or LEGACY_BATCH_ATTENTION_ARCHITECTURE
    declared = source_flow_architecture(config)
    if inferred is not None and inferred != declared:
        raise ValueError(
            "Checkpoint architecture metadata disagrees with its state_dict: "
            f"declared={declared!r}, inferred={inferred!r}."
        )
    config["architecture"] = declared
    return config


class SourceFlowModel(nn.Module):
    """Continuous-time wrapper around the existing conditional stDiff DiT."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        time_embedding_scale: float = 1000.0,
        architecture: str = "custom",
        row_separable: bool = False,
    ) -> None:
        super().__init__()
        if time_embedding_scale <= 0:
            raise ValueError("time_embedding_scale must be positive.")
        self.backbone = backbone
        self.time_embedding_scale = float(time_embedding_scale)
        self.architecture = str(architecture)
        self.row_separable = bool(row_separable)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        t = t.to(device=x.device, dtype=x.dtype)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if tuple(t.shape) != (x.shape[0],):
            raise ValueError(f"t must be scalar or shape [B], got {tuple(t.shape)}.")
        return self.backbone(
            x,
            t=t * self.time_embedding_scale,
            y=condition,
        )


def build_source_flow_model(model_config: Mapping[str, Any]) -> SourceFlowModel:
    """Build the explicitly selected source-flow architecture."""

    architecture = source_flow_architecture(model_config)
    common = {
        "input_size": int(model_config["input_size"]),
        "hidden_size": int(model_config["hidden_size"]),
        "depth": int(model_config.get("depth", 6)),
        "num_heads": int(model_config.get("num_heads", 16)),
        "mlp_ratio": float(model_config.get("mlp_ratio", 4.0)),
        "condition_size": int(
            model_config.get("condition_size", model_config["input_size"])
        ),
    }
    if architecture == GENE_ATTENTION_ARCHITECTURE:
        try:
            from .source_flow_gene_attention import (
                build_gene_attention_source_flow_backbone,
            )
        except ImportError:
            from source_flow_gene_attention import (
                build_gene_attention_source_flow_backbone,
            )

        backbone = build_gene_attention_source_flow_backbone(**common)
        row_separable = True
    else:
        try:
            from .model_original import build_model
        except ImportError:
            from model_original import build_model

        backbone = build_model(
            "stdiff",
            **common,
            classes=int(model_config.get("classes", 6)),
        )
        row_separable = False

    return SourceFlowModel(
        backbone,
        time_embedding_scale=float(model_config.get("time_embedding_scale", 1000.0)),
        architecture=architecture,
        row_separable=row_separable,
    )


def conditional_flow_matching_loss(
    model: SourceFlowModel,
    target: torch.Tensor,
    condition: torch.Tensor,
    *,
    source_noise: Optional[torch.Tensor] = None,
    times: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Straight-path CFM from supplied states (default ``N(0,I)``) to targets."""

    if target.ndim != 2 or condition.ndim != 2:
        raise ValueError("target and condition must both be two-dimensional.")
    if target.shape[0] != condition.shape[0]:
        raise ValueError("target and condition must have the same batch size.")
    if source_noise is None:
        source_noise = torch.randn_like(target)
    if source_noise.shape != target.shape:
        raise ValueError("source_noise must have the same shape as target.")
    if times is None:
        times = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
    if times.ndim == 2 and times.shape[1] == 1:
        times = times[:, 0]
    if tuple(times.shape) != (target.shape[0],):
        raise ValueError("times must have shape [B] or [B,1].")

    interpolation = times.reshape(-1, 1)
    state = (1.0 - interpolation) * source_noise + interpolation * target
    target_velocity = target - source_noise
    predicted_velocity = model(state, times, condition)
    loss = F.mse_loss(predicted_velocity, target_velocity)
    return loss, {
        "source_noise": source_noise,
        "times": times,
        "state": state,
        "target_velocity": target_velocity,
        "predicted_velocity": predicted_velocity,
    }


def _require_torchdiffeq():
    try:
        from torchdiffeq import odeint
    except ImportError as error:
        raise ImportError(
            "Differentiable source-flow sampling requires torchdiffeq. "
            "Install it with `uv pip install torchdiffeq`."
        ) from error
    return odeint


class SourceFlowODESampler(nn.Module):
    """Differentiable integration of the frozen source field via torchdiffeq."""

    def __init__(
        self,
        vector_field: SourceFlowModel,
        *,
        method: str = "rk4",
        num_steps: int = 20,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> None:
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be positive.")
        if rtol <= 0 or atol <= 0:
            raise ValueError("rtol and atol must be positive.")
        self.vector_field = vector_field
        self.method = method
        self.num_steps = int(num_steps)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.architecture = getattr(vector_field, "architecture", "unknown")
        self.row_separable = bool(getattr(vector_field, "row_separable", False))

        # This sampler represents a frozen source field and is always
        # used in inference mode. eval() does not disable autograd.
        self.eval()

    def integrate_between(
        self,
        initial_state: torch.Tensor,
        condition: torch.Tensor,
        *,
        start_time: float,
        end_time: float,
    ) -> torch.Tensor:
        """Continue a state from ``start_time`` to ``end_time``."""

        if not 0.0 <= start_time <= 1.0:
            raise ValueError("start_time must be in [0,1].")
        if not 0.0 <= end_time <= 1.0:
            raise ValueError("end_time must be in [0,1].")
        if end_time < start_time:
            raise ValueError("end_time must be greater than or equal to start_time.")
        if end_time == start_time:
            return initial_state
        odeint = _require_torchdiffeq()
        integration_times = torch.tensor(
            [float(start_time), float(end_time)],
            device=initial_state.device,
            dtype=initial_state.dtype,
        )

        def dynamics(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            return self.vector_field(state, time, condition)

        options = None
        if self.method in {
            "euler",
            "midpoint",
            "rk4",
            "explicit_adams",
            "implicit_adams",
        }:
            options = {"step_size": float(end_time - start_time) / self.num_steps}
        trajectory = odeint(
            dynamics,
            initial_state,
            integration_times,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            options=options,
        )
        return trajectory[-1]

    def integrate_to(
        self,
        initial_state: torch.Tensor,
        condition: torch.Tensor,
        end_time: float,
    ) -> torch.Tensor:
        return self.integrate_between(
            initial_state,
            condition,
            start_time=0.0,
            end_time=end_time,
        )

    def velocity_at(
        self,
        initial_state: torch.Tensor,
        condition: torch.Tensor,
        time: float,
    ) -> torch.Tensor:
        state = self.integrate_to(initial_state, condition, time)
        batch_time = torch.full(
            (state.shape[0],),
            float(time),
            device=state.device,
            dtype=state.dtype,
        )
        return self.vector_field(state, batch_time, condition)

    def forward(
        self, initial_state: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return self.integrate_to(initial_state, condition, 1.0)


@torch.no_grad()
def evaluate_flow_matching_loss(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    seed: int,
    initial_mean: Optional[torch.Tensor] = None,
) -> float:
    """Evaluate CFM with fixed noise/times so epochs remain comparable."""

    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weighted_loss = 0.0
    sample_count = 0
    for batch_index, (target, expression_condition) in enumerate(dataloader, start=1):
        target = target.float().to(device)
        expression_condition = expression_condition.float().to(device)
        epsilon = torch.randn(target.shape, generator=generator, dtype=target.dtype).to(
            device
        )
        times = torch.rand(target.shape[0], generator=generator, dtype=target.dtype).to(
            device
        )
        source_noise = epsilon
        condition = expression_condition
        if initial_mean is not None:
            mean = initial_mean.to(device=device, dtype=target.dtype).reshape(1, -1)
            expanded_mean = mean.expand(target.shape[0], -1)
            source_noise = source_noise + expanded_mean
            condition = torch.cat((expression_condition, expanded_mean), dim=1)
        loss, _ = conditional_flow_matching_loss(
            model,
            target,
            condition,
            source_noise=source_noise,
            times=times,
        )
        batch_loss = float(loss.item())
        if not np.isfinite(batch_loss):
            raise FloatingPointError(
                "Non-finite source-flow validation loss detected in "
                f"batch {batch_index}."
            )
        weighted_loss += batch_loss * target.shape[0]
        sample_count += target.shape[0]
    if was_training:
        model.train()
    if sample_count == 0:
        raise RuntimeError("The validation DataLoader produced no batches.")
    return weighted_loss / sample_count


@dataclass(frozen=True)
class FlowPerformanceResult:
    metrics: dict[str, Any]
    prediction: torch.Tensor


class FlowPerformanceError(ValueError):
    """A checkpoint cannot be used because its recorded performance is invalid."""


def require_flow_performance(
    checkpoint: Mapping[str, Any],
    *,
    label: str,
    performance_key: str = "source_performance",
    expected_evaluation_domain: Optional[str] = "source_validation",
) -> Mapping[str, Any]:
    performance = checkpoint.get(performance_key)
    if not isinstance(performance, Mapping) and performance_key == "source_performance":
        legacy = checkpoint.get("performance")
        legacy_context = legacy.get("context") if isinstance(legacy, Mapping) else None
        if (
            isinstance(legacy, Mapping)
            and isinstance(legacy_context, Mapping)
            and legacy_context.get("evaluation_domain") == "source_validation"
        ):
            performance = legacy
    if not isinstance(performance, Mapping):
        raise FlowPerformanceError(
            f"{label} has no source-only checkpoint-filtering metadata. Retrain "
            "or re-evaluate it with the current source-flow script; legacy target-"
            "spatial performance must not be used for checkpoint filtering."
        )
    context = performance.get("context")
    evaluation_domain = (
        context.get("evaluation_domain") if isinstance(context, Mapping) else None
    )
    if (
        expected_evaluation_domain is not None
        and evaluation_domain != expected_evaluation_domain
    ):
        raise FlowPerformanceError(
            f"{label} performance was evaluated on {evaluation_domain!r}, expected "
            f"{expected_evaluation_domain!r}."
        )
    if performance.get("passed") is not True:
        raise FlowPerformanceError(
            f"{label} failed its performance gate: "
            f"relative_mse={performance.get('relative_mse')}, "
            f"median_SPCC={performance.get('median_SPCC')}."
        )
    thresholds = performance.get("quality_thresholds")
    try:
        validation_loss = float(performance["validation_cfm_loss"])
        relative_mse = float(performance["relative_mse"])
        median_spcc = float(performance["median_SPCC"])
        max_relative_mse = float(thresholds["max_relative_mse"])
        min_median_spcc = float(thresholds["min_median_SPCC"])
    except (KeyError, TypeError, ValueError) as error:
        raise FlowPerformanceError(
            f"{label} has incomplete performance validation metadata."
        ) from error
    if not all(
        np.isfinite(value)
        for value in (
            validation_loss,
            relative_mse,
            median_spcc,
            max_relative_mse,
            min_median_spcc,
        )
    ) or not (relative_mse <= max_relative_mse and median_spcc >= min_median_spcc):
        raise FlowPerformanceError(
            f"{label} has inconsistent performance validation metadata."
        )
    return performance


@torch.no_grad()
def evaluate_flow_imputation(
    model: nn.Module,
    target_expression: torch.Tensor,
    expression_condition: torch.Tensor,
    *,
    target_gene_indices: Sequence[int] | torch.Tensor | np.ndarray,
    baseline_mean: torch.Tensor,
    genes: Sequence[str],
    spot_names: Sequence[str],
    validation_cfm_loss: float,
    device: torch.device,
    batch_size: int,
    num_noise_samples: int,
    seed: int,
    ode_method: str,
    ode_steps: int,
    ode_rtol: float,
    ode_atol: float,
    max_relative_mse: float,
    min_median_spcc: float,
    output_dir: str | Path,
    initial_mean: Optional[torch.Tensor] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    apply_quality_gate: bool = True,
) -> FlowPerformanceResult:
    """Evaluate held-out-gene ODE imputation and a train-mean baseline."""

    if target_expression.ndim != 2 or expression_condition.ndim != 2:
        raise ValueError("Validation expression and condition must be two-dimensional.")
    if target_expression.shape[0] != expression_condition.shape[0]:
        raise ValueError("Validation expression and condition row counts differ.")
    if batch_size < 1 or num_noise_samples < 1 or ode_steps < 1:
        raise ValueError(
            "Validation batch size, noise samples, and ODE steps must be positive."
        )
    if max_relative_mse <= 0:
        raise ValueError("max_relative_mse must be positive.")
    target_indices = torch.as_tensor(target_gene_indices, dtype=torch.long).cpu()
    if target_indices.ndim != 1 or target_indices.numel() < 1:
        raise ValueError("target_gene_indices must be a non-empty vector.")
    if not np.isfinite(validation_cfm_loss):
        raise FlowPerformanceError(
            "Cannot evaluate a source-flow model with a non-finite validation CFM loss."
        )

    expression_cpu = target_expression.detach().float().cpu()
    condition_cpu = expression_condition.detach().float().cpu()
    baseline_cpu = baseline_mean.detach().float().cpu().reshape(-1)
    for label, value in (
        ("target_expression", expression_cpu),
        ("expression_condition", condition_cpu),
        ("baseline_mean", baseline_cpu),
    ):
        if not bool(torch.isfinite(value).all().item()):
            raise FlowPerformanceError(f"{label} contains NaN or Inf values.")
    if baseline_cpu.numel() != expression_cpu.shape[1]:
        raise ValueError("baseline_mean must contain one value per gene.")
    if (
        int(target_indices.min()) < 0
        or int(target_indices.max()) >= expression_cpu.shape[1]
    ):
        raise ValueError("target_gene_indices contains an out-of-range index.")
    if (
        len(genes) != expression_cpu.shape[1]
        or len(spot_names) != expression_cpu.shape[0]
    ):
        raise ValueError("Validation gene or spot labels do not match the tensors.")

    sampler = SourceFlowODESampler(
        model,
        method=ode_method,
        num_steps=ode_steps,
        rtol=ode_rtol,
        atol=ode_atol,
    )
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    prediction_sum = torch.zeros_like(expression_cpu)
    mean_cpu = (
        None
        if initial_mean is None
        else initial_mean.detach().float().cpu().reshape(1, -1)
    )
    if mean_cpu is not None and not bool(torch.isfinite(mean_cpu).all().item()):
        raise FlowPerformanceError("initial_mean contains NaN or Inf values.")
    for _ in range(num_noise_samples):
        for start in range(0, expression_cpu.shape[0], batch_size):
            stop = min(start + batch_size, expression_cpu.shape[0])
            target_batch = expression_cpu[start:stop]
            condition_batch = condition_cpu[start:stop].to(device)
            noise = torch.randn(
                target_batch.shape, generator=generator, dtype=target_batch.dtype
            )
            initial_state = noise
            if mean_cpu is not None:
                expanded_mean = mean_cpu.expand(stop - start, -1)
                initial_state = initial_state + expanded_mean
                condition_batch = torch.cat(
                    (condition_batch, expanded_mean.to(device)), dim=1
                )
            batch_prediction = sampler(initial_state.to(device), condition_batch).cpu()
            if not bool(torch.isfinite(batch_prediction).all().item()):
                raise FlowPerformanceError(
                    "The source-flow ODE solver produced NaN or Inf values. "
                    "The checkpoint is numerically invalid."
                )
            prediction_sum[start:stop] += batch_prediction
    prediction = prediction_sum / num_noise_samples

    truth = ((expression_cpu.index_select(1, target_indices) + 1.0) / 2.0).clamp(0, 1)
    imputed = ((prediction.index_select(1, target_indices) + 1.0) / 2.0).clamp(0, 1)
    baseline = ((baseline_cpu.index_select(0, target_indices) + 1.0) / 2.0).clamp(0, 1)
    baseline = baseline.reshape(1, -1).expand_as(truth)
    mse = float(torch.mean((imputed - truth).square()).item())
    baseline_mse = float(torch.mean((baseline - truth).square()).item())
    if not np.isfinite(mse) or not np.isfinite(baseline_mse):
        raise FlowPerformanceError(
            "Flow-imputation MSE is non-finite; refusing to write invalid metrics."
        )
    relative_mse = mse / baseline_mse if baseline_mse > 0 else float("inf")

    try:
        from .metric import evaluate_prediction
    except ImportError:
        from metric import evaluate_prediction

    selected_genes = [str(genes[index]) for index in target_indices.tolist()]
    destination = Path(output_dir)
    result = evaluate_prediction(
        imputed.numpy(),
        truth.numpy(),
        genes=selected_genes,
        index=[str(value) for value in spot_names],
        include_clustering=False,
        output_dir=destination,
    )
    technical = {
        key: float(value) for key, value in result.summary.iloc[0].to_dict().items()
    }
    median_spcc = technical.get("SPCC", float("nan"))
    passed = (
        bool(
            np.isfinite(validation_cfm_loss)
            and np.isfinite(relative_mse)
            and relative_mse <= max_relative_mse
            and np.isfinite(median_spcc)
            and median_spcc >= min_median_spcc
        )
        if apply_quality_gate
        else None
    )

    def finite_or_none(value: Optional[float]) -> Optional[float]:
        return float(value) if value is not None and np.isfinite(value) else None

    metrics: dict[str, Any] = {
        "passed": passed,
        "quality_gate_applied": apply_quality_gate,
        "validation_cfm_loss": finite_or_none(validation_cfm_loss),
        "mse": mse,
        "baseline_mse": baseline_mse,
        "relative_mse": finite_or_none(relative_mse),
        "median_SPCC": finite_or_none(median_spcc),
        "median_SSIM": finite_or_none(technical.get("SSIM")),
        "median_RMSE": finite_or_none(technical.get("RMSE")),
        "median_JS": finite_or_none(technical.get("JS")),
        "num_evaluation_samples": int(expression_cpu.shape[0]),
        # Retained for compatibility with existing analysis scripts. Consult
        # context.evaluation_domain to distinguish source cells from ST spots.
        "num_validation_spots": int(expression_cpu.shape[0]),
        "num_target_genes": int(target_indices.numel()),
        "num_noise_samples": num_noise_samples,
        "batch_size": min(batch_size, expression_cpu.shape[0]),
        "ode_method": ode_method,
        "ode_steps": ode_steps,
        "quality_thresholds": (
            {
                "max_relative_mse": max_relative_mse,
                "min_median_SPCC": min_median_spcc,
            }
            if apply_quality_gate
            else None
        ),
        "context": dict(metadata or {}),
    }
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "flow_performance.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    torch.save(
        {
            "prediction": prediction,
            "ground_truth": expression_cpu,
            "target_gene_indices": target_indices,
            "baseline_mean": baseline_cpu,
        },
        destination / "flow_performance_tensors.pt",
    )
    return FlowPerformanceResult(metrics=metrics, prediction=prediction)


def freeze_source_flow_model(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return model


def load_source_flow_checkpoint(
    checkpoint_path: str,
    *,
    device: torch.device,
    method: str = "rk4",
    num_steps: int = 20,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    require_row_separable: bool = False,
) -> tuple[SourceFlowODESampler, SourceFlowModel, dict[str, Any]]:
    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError(f"Source-flow checkpoint not found: {source}")

    checkpoint = torch.load(
        source,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("method") != "source_flow":
        raise ValueError(f"Checkpoint is not a source_flow checkpoint: {source}")
    if checkpoint.get("training_status", "complete") != "complete":
        raise FlowPerformanceError(
            f"Source-flow checkpoint evaluation is incomplete: {source}. "
            "Run source_flow_training.py with --evaluation-only for this fold."
        )

    training_settings = checkpoint.get("training_settings")
    if (
        isinstance(training_settings, Mapping)
        and training_settings.get("compile_model") is True
        and training_settings.get("amp_mode") not in (None, "off")
    ):
        numerical_contract = checkpoint.get("numerical_training_contract")
        if not (
            isinstance(numerical_contract, Mapping)
            and numerical_contract.get("compiled_backward_autocast") == "off"
            and numerical_contract.get("loss_finite_check") is True
            and numerical_contract.get("gradient_finite_check") is True
        ):
            raise FlowPerformanceError(
                f"Source-flow checkpoint {source} was trained with AMP plus "
                "torch.compile but lacks the corrected compiled-backward "
                "autocast/numerical contract. Retrain this fold with the current "
                "source_flow_training.py."
            )

    model_config = source_flow_model_config_from_checkpoint(checkpoint)
    architecture = source_flow_architecture(model_config)
    if require_row_separable and architecture != GENE_ATTENTION_ARCHITECTURE:
        raise ValueError(
            "This experiment requires a row-separable source-flow checkpoint, "
            f"but {source} uses {architecture!r}. Retrain it with the current "
            "source_flow_training.py before running the CFM bound experiment."
        )

    source_performance = require_flow_performance(
        checkpoint,
        label=f"Source-flow checkpoint {source}",
    )

    checkpoint["source_performance"] = dict(source_performance)

    model = build_source_flow_model(model_config).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    freeze_source_flow_model(model)
    checkpoint["model_config"] = model_config

    sampler = SourceFlowODESampler(
        model,
        method=method,
        num_steps=num_steps,
        rtol=rtol,
        atol=atol,
    )

    # Source sampler is inference-only, while gradients may still
    # propagate through the ODE w.r.t. its initial state.
    sampler.eval()

    return sampler, model, checkpoint


@dataclass(frozen=True)
class LoadedSourceFlowRun:
    target_expression: torch.Tensor
    observed_mask: torch.Tensor
    missing_mask: torch.Tensor
    sampler: SourceFlowODESampler
    source_model: SourceFlowModel
    checkpoint: dict[str, Any]
    genes: tuple[str, ...]
    spot_names: tuple[str, ...]


def _fold_record(records: list[dict[str, Any]], fold: int) -> dict[str, Any]:
    record = next((item for item in records if int(item["fold"]) == fold), None)
    if record is None:
        raise ValueError(f"Fold {fold} is not present in gene_folds.json.")
    return record


def load_source_flow_run(
    run_dir: str,
    *,
    fold: int,
    device: torch.device,
    sc_path: Optional[str] = None,
    spatial_path: Optional[str] = None,
    ode_method: str = "rk4",
    ode_steps: int = 20,
    ode_rtol: float = 1e-5,
    ode_atol: float = 1e-5,
    require_row_separable: bool = False,
    require_observed_only_preprocessing: bool = False,
) -> LoadedSourceFlowRun:
    """Load one fold plus target ST data using the existing preprocessing path."""

    try:
        from .Dataloader import (
            GENE_LOCAL_PREPROCESSING,
            LEGACY_PREPROCESSING,
            load_paired_h5ad,
            preprocess_pair,
        )
        from .utils import load_json, sha256_file
    except ImportError:
        from Dataloader import (
            GENE_LOCAL_PREPROCESSING,
            LEGACY_PREPROCESSING,
            load_paired_h5ad,
            preprocess_pair,
        )
        from utils import load_json, sha256_file

    root = Path(run_dir).expanduser().resolve()
    experiment_path = root / "experiment.json"
    data_info_path = root / "data_info.json"
    folds_path = root / "gene_folds.json"
    checkpoint_path = root / "checkpoints" / f"fold_{fold:02d}.pt"
    for required in (experiment_path, data_info_path, folds_path, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"Required source-flow artifact not found: {required}"
            )
    experiment = load_json(experiment_path)
    if experiment.get("method") != "source_flow":
        raise ValueError(f"Run is not a source_flow experiment: {root}")
    preprocessing = experiment.get("preprocessing", LEGACY_PREPROCESSING)
    if require_observed_only_preprocessing and preprocessing != GENE_LOCAL_PREPROCESSING:
        raise ValueError(
            "Target observed-only re-masking requires gene-local preprocessing. "
            "Retrain the source flow with the current source_flow_training.py; "
            "legacy checkpoints used full-gene target normalization."
        )
    data_info = load_json(data_info_path)
    records = load_json(folds_path)
    record = _fold_record(records, fold)

    resolved_sc = Path(sc_path or data_info["sc_h5ad"])
    resolved_spatial = Path(spatial_path or data_info["spatial_h5ad"])
    if sha256_file(resolved_sc) != data_info["sc_sha256"]:
        raise ValueError("scRNA H5AD hash differs from source-flow training input.")
    if sha256_file(resolved_spatial) != data_info["spatial_sha256"]:
        raise ValueError("Spatial H5AD hash differs from source-flow training input.")
    adata_seq, adata_spatial = load_paired_h5ad(resolved_sc, resolved_spatial)
    settings = experiment["settings"]
    prepared = preprocess_pair(
        adata_seq,
        adata_spatial,
        method="stdiff",
        noise_std=float(settings["noise_std"]),
        leiden_resolution=float(settings.get("leiden_resolution", 0.4)),
        leiden_random_state=int(settings["split_seed"]),
        preprocessing=preprocessing,
    )
    genes = tuple(str(value) for value in prepared.genes)
    if list(genes) != data_info["genes"]:
        raise ValueError("Gene order differs from source-flow training metadata.")

    target_expression = torch.as_tensor(
        prepared.spatial_array * 2.0 - 1.0,
        dtype=torch.float32,
        device=device,
    )
    missing_indices = np.asarray(record["test_indices"], dtype=np.int64)
    missing_mask = torch.zeros(len(genes), dtype=torch.bool)
    missing_mask[torch.as_tensor(missing_indices)] = True
    observed_mask = ~missing_mask
    sampler, source_model, checkpoint = load_source_flow_checkpoint(
        str(checkpoint_path),
        device=device,
        method=ode_method,
        num_steps=ode_steps,
        rtol=ode_rtol,
        atol=ode_atol,
        require_row_separable=require_row_separable,
    )
    if checkpoint.get("preprocessing", LEGACY_PREPROCESSING) != preprocessing:
        raise ValueError("Checkpoint preprocessing differs from experiment.json.")
    if checkpoint["genes"] != list(genes):
        raise ValueError("Checkpoint gene order differs from target expression.")
    return LoadedSourceFlowRun(
        target_expression=target_expression,
        observed_mask=observed_mask,
        missing_mask=missing_mask,
        sampler=sampler,
        source_model=source_model,
        checkpoint=checkpoint,
        genes=genes,
        spot_names=tuple(str(value) for value in adata_spatial.obs_names),
    )


__all__ = [
    "FlowPerformanceError",
    "FlowPerformanceResult",
    "GENE_ATTENTION_ARCHITECTURE",
    "LEGACY_BATCH_ATTENTION_ARCHITECTURE",
    "LoadedSourceFlowRun",
    "SourceFlowModel",
    "SourceFlowODESampler",
    "build_source_flow_model",
    "conditional_flow_matching_loss",
    "evaluate_flow_imputation",
    "evaluate_flow_matching_loss",
    "freeze_source_flow_model",
    "load_source_flow_checkpoint",
    "load_source_flow_run",
    "require_flow_performance",
    "source_flow_architecture",
    "source_flow_model_config_from_checkpoint",
]
