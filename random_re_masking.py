"""Gene-set handling and observable random re-masking objective.

This module owns the split ``O -> (C, V)`` and the loss on ``V``.  It does not
perform optimizer steps or gradient-alignment analysis.  Conditions keep the
full gene dimension by default for compatibility with the current stDiff data
representation; every gene outside ``C`` is replaced by a fixed value, so its
measured expression cannot leak into the frozen flow.

Smoke Test:
```bash
python random_re_masking.py \
  --source-flow-run-dir ./training_flow_results/dataset5_source_flow_seed0 \
  --fold 1 \
  --remask-ratio 0.2 \
  --condition-fill-value -1.0 \
  --ode-method rk4 \
  --ode-steps 20 \
  --max-spots 32 \
  --kl-weight 0.0001 \
  --seed 0 \
  --device cuda:0 \
  --dtype float32 \
  --output-dir ./random_masking_results/alignment_smoke/fold_01
```

Full Test:
```bash
python random_re_masking.py \
  --source-flow-run-dir ./training_flow_results/dataset5_source_flow_seed0 \
  --fold 0 \
  --spot-batch-size 256 \
  --remask-ratio 0.2 \
  --condition-fill-value -1.0 \
  --ode-method rk4 \
  --ode-steps 20 \
  --kl-weight 0.0001 \
  --seed 0 \
  --device cuda:0 \
  --dtype float32 \
  --output-dir ./random_masking_results/dataset5/
```
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .initialization_adaptation import (
        InitializationAdapter,
        MeanOnlyGaussianInitialization,
        _load_cli_tensor,
        _resolve_cli_device,
    )
except ImportError:  # Support the existing ``python MyCode/script.py`` style.
    from initialization_adaptation import (
        InitializationAdapter,
        MeanOnlyGaussianInitialization,
        _load_cli_tensor,
        _resolve_cli_device,
    )


MaskInput = Union[torch.Tensor, Sequence[bool], Sequence[int], Sequence[float]]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ConditionBuilder = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
FlowSampler = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _as_boolean_mask(values: MaskInput, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values).detach().cpu()
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if tensor.numel() < 1:
        raise ValueError(f"{name} must not be empty.")
    if tensor.dtype != torch.bool:
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf values.")
        if not torch.all((tensor == 0) | (tensor == 1)):
            raise ValueError(f"{name} must contain only boolean or 0/1 values.")
        tensor = tensor.bool()
    return tensor.clone()


def _indices_to_mask(
    indices: Sequence[int], *, num_genes: int, name: str
) -> torch.Tensor:
    if isinstance(num_genes, bool) or not isinstance(num_genes, int):
        raise TypeError("num_genes must be an integer.")
    if num_genes < 1:
        raise ValueError("num_genes must be positive.")
    index_tensor = torch.as_tensor(indices, dtype=torch.long).detach().cpu()
    if index_tensor.ndim != 1 or index_tensor.numel() < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")
    if torch.any(index_tensor < 0) or torch.any(index_tensor >= num_genes):
        raise IndexError(f"{name} contains an index outside [0, {num_genes}).")
    if torch.unique(index_tensor).numel() != index_tensor.numel():
        raise ValueError(f"{name} contains duplicate indices.")
    mask = torch.zeros(num_genes, dtype=torch.bool)
    mask[index_tensor] = True
    return mask


@dataclass(frozen=True)
class GeneSets:
    """Deployment-observed genes ``O`` and oracle-only missing genes ``M``.

    Masks are canonicalized on CPU.  Move them to the expression device only at
    the point of tensor indexing; this also keeps random mask sampling compatible
    with the CPU generators returned by ``MyCode/utils.py::set_seed``.
    """

    observed_mask: torch.Tensor
    missing_mask: torch.Tensor

    def __post_init__(self) -> None:
        observed = _as_boolean_mask(self.observed_mask, name="observed_mask")
        missing = _as_boolean_mask(self.missing_mask, name="missing_mask")
        if observed.shape != missing.shape:
            raise ValueError("observed_mask and missing_mask must have the same shape.")
        if not observed.any():
            raise ValueError("The observed gene set O must not be empty.")
        if not missing.any():
            raise ValueError("The missing gene set M must not be empty.")
        if torch.any(observed & missing):
            raise ValueError("Observed and missing gene sets must be disjoint.")
        object.__setattr__(self, "observed_mask", observed)
        object.__setattr__(self, "missing_mask", missing)

    @classmethod
    def from_masks(cls, observed_mask: MaskInput, missing_mask: MaskInput) -> "GeneSets":
        return cls(
            observed_mask=_as_boolean_mask(observed_mask, name="observed_mask"),
            missing_mask=_as_boolean_mask(missing_mask, name="missing_mask"),
        )

    @classmethod
    def from_indices(
        cls,
        *,
        num_genes: int,
        observed_indices: Sequence[int],
        missing_indices: Sequence[int],
    ) -> "GeneSets":
        return cls(
            observed_mask=_indices_to_mask(
                observed_indices, num_genes=num_genes, name="observed_indices"
            ),
            missing_mask=_indices_to_mask(
                missing_indices, num_genes=num_genes, name="missing_indices"
            ),
        )

    @property
    def num_genes(self) -> int:
        return int(self.observed_mask.numel())

    @property
    def observed_indices(self) -> torch.Tensor:
        return torch.nonzero(self.observed_mask, as_tuple=True)[0]

    @property
    def missing_indices(self) -> torch.Tensor:
        return torch.nonzero(self.missing_mask, as_tuple=True)[0]


@dataclass(frozen=True)
class RemaskSplit:
    """One shared-across-spots partition of observed genes into ``C`` and ``V``."""

    condition_mask: torch.Tensor
    validation_mask: torch.Tensor
    remask_ratio: float

    def __post_init__(self) -> None:
        condition = _as_boolean_mask(self.condition_mask, name="condition_mask")
        validation = _as_boolean_mask(self.validation_mask, name="validation_mask")
        if condition.shape != validation.shape:
            raise ValueError("condition_mask and validation_mask must have the same shape.")
        if not condition.any():
            raise ValueError("The condition gene set C must not be empty.")
        if not validation.any():
            raise ValueError("The held-out gene set V must not be empty.")
        if torch.any(condition & validation):
            raise ValueError("Condition and held-out gene sets must be disjoint.")
        object.__setattr__(self, "condition_mask", condition)
        object.__setattr__(self, "validation_mask", validation)

    @property
    def condition_indices(self) -> torch.Tensor:
        return torch.nonzero(self.condition_mask, as_tuple=True)[0]

    @property
    def validation_indices(self) -> torch.Tensor:
        return torch.nonzero(self.validation_mask, as_tuple=True)[0]

    @property
    def num_condition_genes(self) -> int:
        return int(self.condition_mask.sum().item())

    @property
    def num_remasked_genes(self) -> int:
        return int(self.validation_mask.sum().item())


def sample_remask_split(
    observed_mask: MaskInput,
    remask_ratio: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> RemaskSplit:
    """Sample one gene mask shared by every spot in the batch."""

    observed = _as_boolean_mask(observed_mask, name="observed_mask")
    if not math.isfinite(remask_ratio) or not 0.0 < remask_ratio < 1.0:
        raise ValueError("remask_ratio must be finite and strictly between 0 and 1.")
    observed_indices = torch.nonzero(observed, as_tuple=True)[0]
    num_observed = int(observed_indices.numel())
    if num_observed < 2:
        raise ValueError("At least two observed genes are required for non-empty C and V.")

    requested = int(math.floor(remask_ratio * num_observed + 0.5))
    num_validation = min(max(requested, 1), num_observed - 1)
    permutation_device = torch.device("cpu")
    if generator is not None:
        permutation_device = torch.device(generator.device)
    permutation = torch.randperm(
        num_observed, generator=generator, device=permutation_device
    ).cpu()
    validation_indices = observed_indices[permutation[:num_validation]]
    validation_mask = torch.zeros_like(observed)
    validation_mask[validation_indices] = True
    condition_mask = observed & ~validation_mask
    return RemaskSplit(
        condition_mask=condition_mask,
        validation_mask=validation_mask,
        remask_ratio=float(remask_ratio),
    )


def build_masked_condition(
    expression: torch.Tensor,
    condition_mask: MaskInput,
    *,
    fill_value: float = -1.0,
) -> torch.Tensor:
    """Keep full gene width while replacing every non-condition measurement.

    ``-1`` matches the current ``MyCode/Dataloader.py`` convention after its
    ``[0, 1] -> [-1, 1]`` transformation.  Use ``fill_value=0`` when zero is the
    missing-value convention of another source flow.
    """

    if not isinstance(expression, torch.Tensor) or expression.ndim != 2:
        raise ValueError("expression must be a two-dimensional torch.Tensor.")
    mask = _as_boolean_mask(condition_mask, name="condition_mask")
    if mask.numel() != expression.shape[1]:
        raise ValueError(
            "condition_mask length must equal the expression gene dimension."
        )
    device_mask = mask.to(device=expression.device).reshape(1, -1)
    fill = torch.as_tensor(fill_value, device=expression.device, dtype=expression.dtype)
    return torch.where(device_mask, expression, fill)


def build_compact_condition(
    expression: torch.Tensor, condition_mask: MaskInput
) -> torch.Tensor:
    """Return only ``Y_C`` for flow models that natively accept compact panels."""

    if not isinstance(expression, torch.Tensor) or expression.ndim != 2:
        raise ValueError("expression must be a two-dimensional torch.Tensor.")
    mask = _as_boolean_mask(condition_mask, name="condition_mask")
    if mask.numel() != expression.shape[1]:
        raise ValueError(
            "condition_mask length must equal the expression gene dimension."
        )
    indices = torch.nonzero(mask, as_tuple=True)[0].to(expression.device)
    return expression.index_select(1, indices)


def freeze_source_flow(source_model: Optional[nn.Module]) -> None:
    """Freeze source parameters without disabling gradients with respect to inputs."""

    if source_model is None:
        return
    if not isinstance(source_model, nn.Module):
        raise TypeError("source_model must be a torch.nn.Module or None.")
    source_model.eval()
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


@dataclass(frozen=True)
class RemaskingOutput:
    loss: torch.Tensor
    split: RemaskSplit
    prediction: torch.Tensor
    condition: torch.Tensor


class RandomRemaskingObjective:
    """Construct ``Y_C``, run a differentiable frozen flow, and score only ``V``."""

    def __init__(
        self,
        flow_sampler: FlowSampler,
        initialization_adapter: InitializationAdapter,
        *,
        remask_ratio: float,
        source_model: Optional[nn.Module] = None,
        loss_function: Optional[LossFunction] = None,
        condition_builder: Optional[ConditionBuilder] = None,
        condition_fill_value: float = -1.0,
    ) -> None:
        if not callable(flow_sampler):
            raise TypeError("flow_sampler must be callable.")
        if not isinstance(initialization_adapter, InitializationAdapter):
            raise TypeError("initialization_adapter must implement InitializationAdapter.")
        if not math.isfinite(remask_ratio) or not 0.0 < remask_ratio < 1.0:
            raise ValueError("remask_ratio must be finite and strictly between 0 and 1.")

        self.flow_sampler = flow_sampler
        self.initialization_adapter = initialization_adapter
        self.remask_ratio = float(remask_ratio)
        self.loss_function = F.mse_loss if loss_function is None else loss_function
        self.condition_builder = condition_builder
        self.condition_fill_value = condition_fill_value

        inferred_model = flow_sampler if isinstance(flow_sampler, nn.Module) else None
        freeze_source_flow(inferred_model)
        if source_model is not inferred_model:
            freeze_source_flow(source_model)

    def sample_split(
        self,
        observed_mask: MaskInput,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> RemaskSplit:
        return sample_remask_split(
            observed_mask, self.remask_ratio, generator=generator
        )

    def build_condition(
        self, expression: torch.Tensor, condition_mask: MaskInput
    ) -> torch.Tensor:
        if self.condition_builder is not None:
            condition = self.condition_builder(
                expression, _as_boolean_mask(condition_mask, name="condition_mask")
            )
            if not isinstance(condition, torch.Tensor):
                raise TypeError("condition_builder must return a torch.Tensor.")
            return condition
        return build_masked_condition(
            expression,
            condition_mask,
            fill_value=self.condition_fill_value,
        )

    def loss_for_split(
        self,
        expression: torch.Tensor,
        split: RemaskSplit,
        *,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> RemaskingOutput:
        if not isinstance(expression, torch.Tensor) or expression.ndim != 2:
            raise ValueError("expression must be a two-dimensional torch.Tensor.")
        if expression.shape[1] != split.condition_mask.numel():
            raise ValueError("The remask split does not match the expression genes.")

        condition = self.build_condition(expression, split.condition_mask)
        initial_state = self.initialization_adapter.sample(
            expression.shape[0], noise=noise, generator=generator
        )
        prediction = self.flow_sampler(initial_state, condition)
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("flow_sampler must return a torch.Tensor.")
        if prediction.shape != expression.shape:
            raise ValueError(
                "flow_sampler output must have the same shape as expression; "
                f"got {tuple(prediction.shape)} and {tuple(expression.shape)}."
            )

        validation_mask = split.validation_mask.to(expression.device)
        loss = self.loss_function(
            prediction[:, validation_mask], expression[:, validation_mask]
        )
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError("loss_function must return a scalar torch.Tensor.")
        return RemaskingOutput(
            loss=loss,
            split=split,
            prediction=prediction,
            condition=condition,
        )

    def __call__(
        self,
        expression: torch.Tensor,
        observed_mask: MaskInput,
        *,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> RemaskingOutput:
        split = self.sample_split(observed_mask, generator=generator)
        return self.loss_for_split(
            expression, split, noise=noise, generator=generator
        )


def adaptation_objective(
    remask_loss: torch.Tensor,
    initialization_adapter: InitializationAdapter,
    *,
    kl_weight: float = 0.0,
) -> torch.Tensor:
    """Add KL regularization for an actual adaptation update, not alignment."""

    if not math.isfinite(kl_weight) or kl_weight < 0:
        raise ValueError("kl_weight must be finite and non-negative.")
    return remask_loss + kl_weight * initialization_adapter.regularization()


def _load_flow_for_cli(
    *,
    flow_model: Optional[str],
    flow_factory: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[FlowSampler, Optional[nn.Module]]:
    if (flow_model is None) == (flow_factory is None):
        raise ValueError("Provide exactly one of --flow-model or --flow-factory.")

    if flow_model is not None:
        source = Path(flow_model)
        if not source.is_file():
            raise FileNotFoundError(f"TorchScript flow sampler not found: {source}")
        model = torch.jit.load(str(source), map_location=device)
        model.to(device=device, dtype=dtype)
        return model, model

    module_name, separator, function_name = str(flow_factory).rpartition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("--flow-factory must use MODULE:FUNCTION or FILE.py:FUNCTION.")
    module_path = Path(module_name)
    if module_path.suffix.lower() == ".py" or module_path.is_file():
        if not module_path.is_file():
            raise FileNotFoundError(f"Flow factory module not found: {module_path}")
        spec = importlib.util.spec_from_file_location(
            "mycode_flow_factory", module_path.resolve()
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import flow factory module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise AttributeError(f"Flow factory is not callable: {flow_factory}")
    built = factory(device=device, dtype=dtype)

    source_model: Optional[nn.Module] = None
    if isinstance(built, dict):
        flow_sampler = built.get("flow_sampler")
        source_model = built.get("source_model")
    elif isinstance(built, tuple) and len(built) == 2:
        flow_sampler, source_model = built
    else:
        flow_sampler = built
        if isinstance(built, nn.Module):
            source_model = built
    if not callable(flow_sampler):
        raise TypeError("Flow factory must return a callable flow_sampler.")
    if source_model is not None and not isinstance(source_model, nn.Module):
        raise TypeError("source_model returned by the factory must be nn.Module or None.")
    if source_model is not None:
        source_model.to(device=device, dtype=dtype)
    return flow_sampler, source_model


def _load_expression_for_cli(
    path: str, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    expression = _load_cli_tensor(
        path, key="target_expression", device=device, dtype=dtype
    )
    if expression.ndim != 2 or expression.shape[0] < 1 or expression.shape[1] < 1:
        raise ValueError("Expression tensor must have shape [B,G] with B,G > 0.")
    if not torch.isfinite(expression).all():
        raise ValueError("Expression tensor contains NaN or Inf values.")
    return expression


def _load_mask_for_cli(path: str, *, num_genes: int, name: str) -> torch.Tensor:
    mask = _load_cli_tensor(
        path,
        key=name,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    mask = _as_boolean_mask(mask, name=name)
    if mask.numel() != num_genes:
        raise ValueError(f"{name} must contain {num_genes} values.")
    return mask


def _load_fold_masks_for_cli(
    path: str, *, fold: int, num_genes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"gene_folds.json not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if isinstance(records, dict):
        records = records.get("folds", records.get("gene_folds", records))
    if not isinstance(records, list):
        raise ValueError("gene_folds.json must contain a list of fold records.")
    record = next(
        (
            candidate
            for candidate in records
            if int(candidate.get("fold", -1)) == fold
        ),
        None,
    )
    if record is None:
        raise ValueError(f"Fold {fold} is not present in {source}.")
    missing_indices = record.get("test_indices")
    if missing_indices is None:
        raise KeyError(f"Fold {fold} does not contain test_indices.")
    missing = _indices_to_mask(
        missing_indices, num_genes=num_genes, name="test_indices"
    )
    return ~missing, missing


def _available_fold_numbers_for_cli(
    *,
    source_flow_run_dir: Optional[str],
    gene_folds: Optional[str],
) -> list[int]:
    if source_flow_run_dir is not None:
        folds_path = Path(source_flow_run_dir) / "gene_folds.json"
    elif gene_folds is not None:
        folds_path = Path(gene_folds)
    else:
        raise ValueError(
            "--fold 0 requires --source-flow-run-dir or --gene-folds; "
            "explicit masks define only one experiment."
        )
    if not folds_path.is_file():
        raise FileNotFoundError(f"gene_folds.json not found: {folds_path}")
    with folds_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if isinstance(records, dict):
        records = records.get("folds", records.get("gene_folds", records))
    if not isinstance(records, list):
        raise ValueError("gene_folds.json must contain a list of fold records.")
    folds = sorted({int(record["fold"]) for record in records})
    if not folds or folds[0] < 1:
        raise ValueError("gene_folds.json must contain positive fold numbers.")
    if source_flow_run_dir is not None:
        checkpoint_dir = Path(source_flow_run_dir) / "checkpoints"
        missing = [
            fold
            for fold in folds
            if not (checkpoint_dir / f"fold_{fold:02d}.pt").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Source-flow checkpoints are missing for folds: {missing}."
            )
    return folds


def _prepare_output_dir(path: str, *, allow_existing: bool) -> Path:
    output_dir = Path(path)
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new directory or pass --allow-existing."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _spot_batch_slices(
    num_spots: int, spot_batch_size: Optional[int]
) -> tuple[tuple[int, int], ...]:
    if isinstance(num_spots, bool) or not isinstance(num_spots, int):
        raise TypeError("num_spots must be an integer.")
    if num_spots < 1:
        raise ValueError("num_spots must be positive.")
    if spot_batch_size is None:
        batch_size = num_spots
    else:
        if isinstance(spot_batch_size, bool) or not isinstance(spot_batch_size, int):
            raise TypeError("spot_batch_size must be an integer or None.")
        if spot_batch_size < 1:
            raise ValueError("spot_batch_size must be positive.")
        batch_size = min(spot_batch_size, num_spots)
    return tuple(
        (start, min(start + batch_size, num_spots))
        for start in range(0, num_spots, batch_size)
    )


def _build_adapter_for_cli(
    *,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
    mu_path: Optional[str],
) -> MeanOnlyGaussianInitialization:
    adapter = MeanOnlyGaussianInitialization(
        dimension, device=device, dtype=dtype
    )
    if mu_path is not None:
        mu = _load_cli_tensor(mu_path, key="mu", device=device, dtype=dtype)
        if mu.ndim == 1:
            mu = mu.unsqueeze(0)
        if tuple(mu.shape) != tuple(adapter.mu.shape):
            raise ValueError(
                f"mu must have shape {tuple(adapter.mu.shape)}, got {tuple(mu.shape)}."
            )
        with torch.no_grad():
            adapter.mu.copy_(mu)
    return adapter


def _load_single_noise_for_cli(
    path: Optional[str],
    *,
    adapter: InitializationAdapter,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if path is None:
        return adapter.sample_noise(batch_size, generator=generator)
    parameter = adapter.adaptation_parameters()[0]
    noise = _load_cli_tensor(
        path,
        key="noise_samples",
        device=parameter.device,
        dtype=parameter.dtype,
    )
    if noise.ndim == 3 and noise.shape[0] == 1:
        noise = noise[0]
    expected = (batch_size, adapter.dimension)
    if tuple(noise.shape) != expected:
        raise ValueError(f"noise must have shape {expected}, got {tuple(noise.shape)}.")
    return noise


def _add_flow_cli_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--source-flow-run-dir",
        help="Source-flow training run; loads checkpoint, target data, and fold masks.",
    )
    group.add_argument(
        "--flow-model",
        help="TorchScript sampler whose forward(X0, condition) returns [B,G].",
    )
    group.add_argument(
        "--flow-factory",
        help="MODULE:FUNCTION or FILE.py:FUNCTION returning the flow sampler.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one observable random re-masking loss."
    )
    parser.add_argument("--expression", default=None, help="Preprocessed [B,G] .pt/.npy tensor.")
    mask_group = parser.add_mutually_exclusive_group(required=False)
    mask_group.add_argument("--observed-mask", help="Observed [G] 0/1 .pt/.npy mask.")
    mask_group.add_argument("--gene-folds", help="Existing gene_folds.json file.")
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold number; use 0 to run every fold into fold_XX subdirectories.",
    )
    _add_flow_cli_arguments(parser)
    parser.add_argument("--sc-path", default=None)
    parser.add_argument("--spatial-path", default=None)
    parser.add_argument("--ode-method", default="rk4")
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--ode-rtol", type=float, default=1e-5)
    parser.add_argument("--ode-atol", type=float, default=1e-5)
    parser.add_argument(
        "--max-spots",
        type=int,
        default=None,
        help="Use only the first N target spots; intended for subset smoke tests.",
    )
    parser.add_argument(
        "--spot-batch-size",
        type=int,
        default=32,
        help="Process all selected spots in batches of this size (default: 32).",
    )
    parser.add_argument("--initialization-dim", type=int, default=None)
    parser.add_argument("--mu", default=None, help="Optional [d] or [1,d] .pt/.npy tensor.")
    parser.add_argument("--noise", default=None, help="Optional [B,d] .pt/.npy tensor.")
    parser.add_argument("--remask-ratio", type=float, required=True)
    parser.add_argument("--condition-fill-value", type=float, default=-1.0)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1202)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-existing", action="store_true")
    return parser


def _run_single_fold(args: argparse.Namespace) -> dict[str, Any]:
    device = _resolve_cli_device(args.device)
    dtype = getattr(torch, args.dtype)
    source_flow_performance = None
    source_flow_target_transfer_performance = None
    if args.source_flow_run_dir is not None:
        if (
            args.expression is not None
            or args.gene_folds is not None
            or args.observed_mask is not None
        ):
            raise ValueError(
                "--source-flow-run-dir already supplies expression and masks; "
                "do not combine it with --expression/--gene-folds/--observed-mask."
            )
        if dtype != torch.float32:
            raise ValueError("Source-flow checkpoints currently require --dtype float32.")
        try:
            from .source_flow import load_source_flow_run
        except ImportError:
            from source_flow import load_source_flow_run

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
            require_row_separable=True,
            require_observed_only_preprocessing=True,
        )
        expression = loaded.target_expression.cpu()
        observed_mask = loaded.observed_mask
        flow_sampler = loaded.sampler
        source_model = loaded.source_model
        source_flow_performance = dict(loaded.checkpoint["source_performance"])
        target_transfer = loaded.checkpoint.get("target_transfer_performance")
        if isinstance(target_transfer, dict):
            source_flow_target_transfer_performance = dict(target_transfer)
        del loaded
    else:
        if args.expression is None:
            raise ValueError("--expression is required without --source-flow-run-dir.")
        expression = _load_expression_for_cli(
            args.expression, device=torch.device("cpu"), dtype=dtype
        )
        if args.gene_folds is not None:
            observed_mask, _ = _load_fold_masks_for_cli(
                args.gene_folds, fold=args.fold, num_genes=expression.shape[1]
            )
        elif args.observed_mask is not None:
            observed_mask = _load_mask_for_cli(
                args.observed_mask,
                num_genes=expression.shape[1],
                name="observed_mask",
            )
        else:
            raise ValueError(
                "Provide --gene-folds or --observed-mask without --source-flow-run-dir."
            )
        flow_sampler, source_model = _load_flow_for_cli(
            flow_model=args.flow_model,
            flow_factory=args.flow_factory,
            device=device,
            dtype=dtype,
        )

    if args.max_spots is not None:
        if args.max_spots < 1:
            raise ValueError("max_spots must be positive.")
        expression = expression[: args.max_spots]

    dimension = expression.shape[1] if args.initialization_dim is None else args.initialization_dim
    adapter = _build_adapter_for_cli(
        dimension=dimension,
        device=device,
        dtype=dtype,
        mu_path=args.mu,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = _load_single_noise_for_cli(
        args.noise,
        adapter=adapter,
        batch_size=expression.shape[0],
        generator=generator,
    ).cpu()
    objective = RandomRemaskingObjective(
        flow_sampler,
        adapter,
        remask_ratio=args.remask_ratio,
        source_model=source_model,
        condition_fill_value=args.condition_fill_value,
    )
    split = objective.sample_split(observed_mask, generator=generator)
    spot_batches = _spot_batch_slices(expression.shape[0], args.spot_batch_size)
    weighted_loss = 0.0
    condition_parts = []
    prediction_parts = []
    with torch.no_grad():
        for start, stop in spot_batches:
            batch_expression = expression[start:stop].to(device=device, dtype=dtype)
            batch_output = objective.loss_for_split(
                batch_expression,
                split,
                noise=noise[start:stop],
            )
            weighted_loss += float(batch_output.loss.item()) * (stop - start)
            condition_parts.append(batch_output.condition.cpu())
            prediction_parts.append(batch_output.prediction.cpu())
        remask_loss = torch.as_tensor(
            weighted_loss / expression.shape[0], device=device, dtype=dtype
        )
        total_loss = adaptation_objective(
            remask_loss, adapter, kl_weight=args.kl_weight
        )
    condition = torch.cat(condition_parts, dim=0)
    prediction = torch.cat(prediction_parts, dim=0)
    metrics: dict[str, Any] = {
        "remask_loss": float(remask_loss.item()),
        "kl_regularization": float(adapter.regularization().detach().item()),
        "adaptation_loss": float(total_loss.detach().item()),
        "num_condition_genes": split.num_condition_genes,
        "num_remasked_genes": split.num_remasked_genes,
        "remask_ratio": args.remask_ratio,
        "kl_weight": args.kl_weight,
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "source_flow_run_dir": (
            None
            if args.source_flow_run_dir is None
            else str(Path(args.source_flow_run_dir).expanduser().resolve())
        ),
        "source_flow_performance": source_flow_performance,
        "source_flow_target_transfer_performance": (
            source_flow_target_transfer_performance
        ),
        "fold": args.fold if args.source_flow_run_dir or args.gene_folds else None,
        "ode_method": args.ode_method if args.source_flow_run_dir else None,
        "ode_steps": args.ode_steps if args.source_flow_run_dir else None,
        "ode_rtol": args.ode_rtol if args.source_flow_run_dir else None,
        "ode_atol": args.ode_atol if args.source_flow_run_dir else None,
        "num_spots": int(expression.shape[0]),
        "spot_batch_size": max(stop - start for start, stop in spot_batches),
        "num_spot_batches": len(spot_batches),
    }
    output_dir = _prepare_output_dir(
        args.output_dir, allow_existing=args.allow_existing
    )
    with (output_dir / "remask_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    torch.save(
        {
            "mu": adapter.mu.detach().cpu(),
            "noise": noise,
            "condition_mask": split.condition_mask,
            "validation_mask": split.validation_mask,
            "condition": condition,
            "prediction": prediction,
        },
        output_dir / "remask_tensors.pt",
    )
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.fold < 0:
        raise ValueError("--fold must be 0 or a positive fold number.")
    if args.spot_batch_size < 1:
        raise ValueError("--spot-batch-size must be positive.")
    if args.fold > 0:
        metrics = _run_single_fold(args)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    folds = _available_fold_numbers_for_cli(
        source_flow_run_dir=args.source_flow_run_dir,
        gene_folds=args.gene_folds,
    )
    root = _prepare_output_dir(
        args.output_dir, allow_existing=args.allow_existing
    )
    try:
        from .source_flow import FlowPerformanceError
    except ImportError:
        from source_flow import FlowPerformanceError

    fold_results = {}
    completed_folds = []
    skipped_folds = []
    for fold in folds:
        fold_args = argparse.Namespace(**vars(args))
        fold_args.fold = fold
        fold_args.output_dir = str(root / f"fold_{fold:02d}")
        key = f"fold_{fold:02d}"
        try:
            metrics = _run_single_fold(fold_args)
        except FlowPerformanceError as error:
            skipped_folds.append(fold)
            fold_results[key] = {
                "status": "skipped",
                "output_dir": fold_args.output_dir,
                "reason": str(error),
            }
            print(f"Skipping {key}: {error}", file=sys.stderr)
            continue
        completed_folds.append(fold)
        fold_results[key] = {
            "status": "completed",
            "output_dir": fold_args.output_dir,
            "metrics": metrics,
        }
    summary = {
        "folds": folds,
        "completed_folds": completed_folds,
        "skipped_folds": skipped_folds,
        "output_dir": str(root),
        "results": fold_results,
    }
    with (root / "random_remasking_all_folds.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


__all__ = [
    "ConditionBuilder",
    "FlowSampler",
    "GeneSets",
    "LossFunction",
    "RandomRemaskingObjective",
    "RemaskingOutput",
    "RemaskSplit",
    "adaptation_objective",
    "build_compact_condition",
    "build_masked_condition",
    "freeze_source_flow",
    "sample_remask_split",
]


if __name__ == "__main__":
    main()
