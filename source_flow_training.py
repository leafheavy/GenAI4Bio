"""Train row-separable gene-attention source flow-matching models."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from .source_flow import (
        GENE_ATTENTION_ARCHITECTURE,
        build_source_flow_model,
        conditional_flow_matching_loss,
        evaluate_flow_imputation,
        evaluate_flow_matching_loss,
        freeze_source_flow_model,
        source_flow_model_config_from_checkpoint,
    )
except ImportError:
    from source_flow import (
        GENE_ATTENTION_ARCHITECTURE,
        build_source_flow_model,
        conditional_flow_matching_loss,
        evaluate_flow_imputation,
        evaluate_flow_matching_loss,
        freeze_source_flow_model,
        source_flow_model_config_from_checkpoint,
    )


def resolve_amp_dtype(
    device: torch.device,
    amp_mode: str = "auto",
) -> torch.dtype | None:
    if amp_mode == "off":
        return None

    if device.type != "cuda":
        return None

    if amp_mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if amp_mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 CUDA 设备不支持 BF16。")
        return torch.bfloat16

    if amp_mode == "fp16":
        return torch.float16

    raise ValueError(f"Unsupported AMP mode: {amp_mode}")


def _configure_compiled_backward_autocast(
    *,
    compile_model: bool,
    amp_dtype: torch.dtype | None,
) -> str:
    """Make the compiled backward match this loop's autocast placement.

    The forward is evaluated inside autocast and ``backward()`` is intentionally
    called outside it, following PyTorch AMP guidance.  Compiled autograd must
    therefore be told not to apply autocast to the backward pass.  PyTorch warns
    that leaving its default forward-context assumption in this pattern can
    silently produce incorrect gradients.
    """

    if not compile_model or amp_dtype is None:
        return "not_applicable"
    try:
        from torch._functorch import config as functorch_config
    except ImportError as error:
        raise RuntimeError(
            "torch.compile together with AMP requires a PyTorch version that "
            "provides torch._functorch.config.backward_pass_autocast. Upgrade "
            "PyTorch or omit --torch-compile."
        ) from error
    if not hasattr(functorch_config, "backward_pass_autocast"):
        raise RuntimeError(
            "This PyTorch version cannot configure compiled-backward autocast. "
            "Upgrade PyTorch or omit --torch-compile when AMP is enabled."
        )
    functorch_config.backward_pass_autocast = "off"
    return "off"


def _atomic_torch_save(payload: object, destination: str | Path) -> Path:
    """Write a checkpoint without exposing a partially written final file."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


@dataclass(frozen=True)
class SourceFlowTrainSettings:
    batch_size: int = 512
    hidden_size: int = 1024
    depth: int = 6
    num_heads: int = 16
    mlp_ratio: float = 4.0
    epochs: int = 900
    learning_rate: float = 1e-4
    noise_std: float = 10.0
    n_splits: int = 5
    split_seed: int = 0
    model_seed: int = 1202
    time_embedding_scale: float = 1000.0
    num_workers: int = 0
    validation_fraction: float = 0.2
    validation_batch_size: int = 32
    validation_max_samples: int = 512
    validation_num_noise_samples: int = 4
    validation_ode_steps: int = 20
    max_validation_relative_mse: float = 1.0
    min_validation_spcc: float = 0.0
    fail_on_performance: bool = True
    show_progress: bool = True
    deterministic: bool = False
    amp_mode: str = "auto"
    compile_model: bool = False
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0

    def validate(self) -> None:
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive.")
        if self.hidden_size < 1 or self.depth < 1 or self.num_heads < 1:
            raise ValueError("Model dimensions must be positive.")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if self.learning_rate <= 0 or self.time_embedding_scale <= 0:
            raise ValueError("learning_rate and time_embedding_scale must be positive.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between 0 and 1.")
        if (
            min(
                self.validation_batch_size,
                self.validation_max_samples,
                self.validation_num_noise_samples,
                self.validation_ode_steps,
            )
            < 1
        ):
            raise ValueError(
                "Validation sizes, samples, and ODE steps must be positive."
            )
        if self.max_validation_relative_mse <= 0:
            raise ValueError("max_validation_relative_mse must be positive.")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative.")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative.")


def train_one_source_flow_fold(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    learning_rate: float,
    epochs: int,
    device: torch.device,
    show_progress: bool,
    validation_dataloader: Optional[torch.utils.data.DataLoader] = None,
    validation_seed: int = 0,
    amp_mode: str = "auto",
    compile_model: bool = False,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    history_callback: Optional[Callable[[Sequence[Mapping[str, float]]], None]] = None,
) -> tuple[list[dict[str, float]], torch.optim.Optimizer]:
    """Train straight-path conditional flow matching on source scRNA only."""

    model.to(device)
    model.train()

    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative.")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta cannot be negative.")
    if early_stopping_patience and validation_dataloader is None:
        raise ValueError("Early stopping requires a validation DataLoader.")

    amp_dtype = resolve_amp_dtype(device, amp_mode)
    use_amp = amp_dtype is not None
    use_fp16_scaler = amp_dtype == torch.float16
    _configure_compiled_backward_autocast(
        compile_model=compile_model,
        amp_dtype=amp_dtype,
    )

    train_model = model
    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile requires PyTorch 2.x.")
        train_model = torch.compile(
            model,
            mode="default",
            fullgraph=False,
            dynamic=None,
        )

    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=0.0)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_fp16_scaler,
    )

    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_model_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    iterator = (
        tqdm(range(epochs), ncols=100, desc="train source flow")
        if show_progress
        else range(epochs)
    )
    for epoch in iterator:
        total_loss = 0.0
        batch_count = 0
        for batch_index, (target, condition) in enumerate(dataloader, start=1):
            target = target.float().to(device)
            condition = condition.float().to(device)
            optimizer.zero_grad(set_to_none=True)

            # Use automatic mixed precision if enabled
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                loss, _ = conditional_flow_matching_loss(
                    train_model,
                    target,
                    condition,
                )

            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    "Non-finite source-flow loss detected at "
                    f"epoch {epoch + 1}, batch {batch_index}. Training stopped "
                    "before the optimizer could corrupt the checkpoint."
                )

            scaler.scale(loss).backward()

            # 梯度裁剪必须在 unscale 之后
            scaler.unscale_(optimizer)
            try:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                raise FloatingPointError(
                    "Non-finite source-flow gradient detected at "
                    f"epoch {epoch + 1}, batch {batch_index}."
                ) from error

            scaler.step(optimizer)
            scaler.update()

            # # Traditional optimizer updating
            # optimizer.zero_grad(set_to_none=True)
            # loss, _ = conditional_flow_matching_loss(model, target, condition)
            # loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # optimizer.step()

            total_loss += float(loss.detach().item())
            batch_count += 1
        if batch_count == 0:
            raise RuntimeError("The source-flow DataLoader produced no batches.")
        nonfinite_parameters = [
            name
            for name, parameter in model.named_parameters()
            if not bool(torch.isfinite(parameter).all().item())
        ]
        if nonfinite_parameters:
            preview = ", ".join(nonfinite_parameters[:3])
            raise FloatingPointError(
                "Non-finite source-flow parameters detected after "
                f"epoch {epoch + 1}: {preview}."
            )
        mean_loss = total_loss / batch_count
        record = {"epoch": float(epoch + 1), "flow_matching_loss": mean_loss}
        if validation_dataloader is not None:
            validation_loss = evaluate_flow_matching_loss(
                model,
                validation_dataloader,
                device=device,
                seed=validation_seed,
            )
            if not np.isfinite(validation_loss):
                raise FloatingPointError(
                    "Non-finite source-flow validation loss detected after "
                    f"epoch {epoch + 1}."
                )
            record["validation_flow_matching_loss"] = validation_loss

            if early_stopping_patience:
                if validation_loss < (best_validation_loss - early_stopping_min_delta):
                    best_validation_loss = validation_loss
                    best_model_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
        history.append(record)
        if history_callback is not None:
            history_callback(history)
        if show_progress:
            validation_text = (
                ""
                if validation_dataloader is None
                else f", val:{record['validation_flow_matching_loss']:.6f}"
            )
            iterator.set_postfix_str(f"cfm loss:{mean_loss:.6f}{validation_text}")
        if (
            early_stopping_patience
            and epochs_without_improvement >= early_stopping_patience
        ):
            break

    if early_stopping_patience:
        if best_model_state is None:
            raise RuntimeError("Early stopping did not observe a valid checkpoint.")
        model.load_state_dict(best_model_state, strict=True)
        if best_epoch != len(history):
            # Adam moments belong to the last optimized epoch, not the restored
            # best epoch.  Clearing them is safer than serializing a mismatched
            # optimizer state.
            optimizer.state.clear()
    else:
        best_epoch = len(history)
    for record in history:
        record["selected_checkpoint"] = float(int(record["epoch"]) == best_epoch)
    if history_callback is not None:
        history_callback(history)
    return history, optimizer


def _split_source_cells(
    num_cells: int,
    validation_fraction: float,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the original/augmented copy of each source cell in one split."""

    if num_cells < 4 or num_cells % 2:
        raise ValueError("Augmented source data must contain paired cells.")
    original_count = num_cells // 2
    validation_count = min(
        original_count - 1,
        max(1, int(round(original_count * validation_fraction))),
    )
    order = torch.randperm(original_count, generator=generator)
    validation_base = order[:validation_count]
    training_base = order[validation_count:]
    return (
        torch.cat((training_base, training_base + original_count)),
        torch.cat((validation_base, validation_base + original_count)),
    )


def _resolve_input_paths(
    dataset: str,
    data_root: str,
    sc_path: Optional[str],
    spatial_path: Optional[str],
):
    try:
        from .Dataloader import resolve_dataset_paths
    except ImportError:
        from Dataloader import resolve_dataset_paths

    if (sc_path is None) != (spatial_path is None):
        raise ValueError("--sc-path and --spatial-path must be provided together.")
    resolved = resolve_dataset_paths(dataset, data_root)
    if sc_path is not None:
        return Path(sc_path), Path(spatial_path), resolved.dataset
    return resolved.sc_h5ad, resolved.spatial_h5ad, resolved.dataset


def _is_source_only_performance(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    context = value.get("context")
    return bool(
        isinstance(context, Mapping)
        and context.get("evaluation_domain") == "source_validation"
        and context.get("evaluation_role") == "checkpoint_filter"
        and value.get("quality_gate_applied") is True
        and isinstance(value.get("passed"), bool)
    )


def _is_target_report_only_performance(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    context = value.get("context")
    return bool(
        isinstance(context, Mapping)
        and context.get("evaluation_domain") == "target_spatial"
        and context.get("evaluation_role") == "report_only"
        and value.get("quality_gate_applied") is False
        and value.get("passed") is None
    )


def _validated_saved_indices(
    value: object,
    *,
    name: str,
    upper_bound: int,
) -> torch.Tensor:
    if value is None:
        raise ValueError(f"Checkpoint is missing saved {name}.")
    indices = torch.as_tensor(value, dtype=torch.long).cpu().reshape(-1)
    if indices.numel() < 1:
        raise ValueError(f"Checkpoint {name} must be non-empty.")
    if int(indices.min()) < 0 or int(indices.max()) >= upper_bound:
        raise ValueError(f"Checkpoint {name} contains out-of-range indices.")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError(f"Checkpoint {name} contains duplicate indices.")
    return indices


def reevaluate_source_flow_run(
    run_dir: str,
    *,
    device: Optional[str] = None,
    fold: Optional[int] = None,
    sc_path: Optional[str] = None,
    spatial_path: Optional[str] = None,
    force_evaluation: bool = False,
    fail_on_performance: bool = True,
) -> Path:
    """Add source-only filtering and target report-only metrics to old weights.

    The saved model and optimizer states are never trained or modified.  The
    checkpoint file is atomically rewritten only to attach evaluation metadata.
    """

    try:
        from .Dataloader import (
            LEGACY_PREPROCESSING,
            GeneFold,
            build_fold_data,
            get_data_loader,
            load_paired_h5ad,
            preprocess_pair,
        )
        from .utils import load_json, resolve_device, save_json, sha256_file
    except ImportError:
        from Dataloader import (
            LEGACY_PREPROCESSING,
            GeneFold,
            build_fold_data,
            get_data_loader,
            load_paired_h5ad,
            preprocess_pair,
        )
        from utils import load_json, resolve_device, save_json, sha256_file

    root = Path(run_dir).expanduser().resolve()
    experiment_path = root / "experiment.json"
    data_info_path = root / "data_info.json"
    folds_path = root / "gene_folds.json"
    checkpoint_dir = root / "checkpoints"
    for required in (experiment_path, data_info_path, folds_path, checkpoint_dir):
        if not required.exists():
            raise FileNotFoundError(
                f"Required source-flow artifact not found: {required}"
            )

    experiment = load_json(experiment_path)
    data_info = load_json(data_info_path)
    fold_records = load_json(folds_path)
    if experiment.get("method") != "source_flow":
        raise ValueError(f"Run is not a source_flow experiment: {root}")
    if not isinstance(fold_records, list) or not fold_records:
        raise ValueError("gene_folds.json must contain a non-empty list.")
    if (sc_path is None) != (spatial_path is None):
        raise ValueError("--sc-path and --spatial-path must be provided together.")

    resolved_sc = Path(sc_path or data_info["sc_h5ad"]).expanduser()
    resolved_spatial = Path(spatial_path or data_info["spatial_h5ad"]).expanduser()
    if sha256_file(resolved_sc) != data_info["sc_sha256"]:
        raise ValueError("scRNA H5AD hash differs from the original training input.")
    if sha256_file(resolved_spatial) != data_info["spatial_sha256"]:
        raise ValueError("Spatial H5AD hash differs from the original training input.")

    run_settings = experiment.get("settings")
    if not isinstance(run_settings, Mapping):
        raise ValueError("experiment.json has no valid source-flow settings.")
    defaults = SourceFlowTrainSettings()
    resolved_device = resolve_device(device)
    adata_seq, adata_spatial = load_paired_h5ad(resolved_sc, resolved_spatial)
    prepared = preprocess_pair(
        adata_seq,
        adata_spatial,
        method="stdiff",
        noise_std=float(run_settings.get("noise_std", defaults.noise_std)),
        leiden_random_state=int(run_settings.get("split_seed", defaults.split_seed)),
        preprocessing=experiment.get("preprocessing", LEGACY_PREPROCESSING),
    )
    genes = [str(value) for value in prepared.genes]
    if genes != list(data_info["genes"]):
        raise ValueError("Gene order differs from the original source-flow run.")

    selected_records = [
        record
        for record in fold_records
        if fold is None or int(record["fold"]) == int(fold)
    ]
    if not selected_records:
        raise ValueError(f"fold must identify one of the saved folds, got {fold}.")

    processed_folds: list[int] = []
    skipped_folds: list[int] = []
    failed_folds: list[int] = []
    for record in selected_records:
        fold_number = int(record["fold"])
        checkpoint_path = checkpoint_dir / f"fold_{fold_number:02d}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Source-flow checkpoint not found: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("method") != "source_flow":
            raise ValueError(f"Checkpoint is not source_flow: {checkpoint_path}")
        if int(checkpoint.get("fold", -1)) != fold_number:
            raise ValueError(f"Checkpoint fold mismatch: {checkpoint_path}")
        if checkpoint.get("preprocessing", LEGACY_PREPROCESSING) != experiment.get(
            "preprocessing", LEGACY_PREPROCESSING
        ):
            raise ValueError(f"Checkpoint preprocessing mismatch: {checkpoint_path}")
        if list(checkpoint.get("genes", [])) != genes:
            raise ValueError(f"Checkpoint gene order mismatch: {checkpoint_path}")

        source_performance = checkpoint.get("source_performance")
        target_performance = checkpoint.get("target_transfer_performance")
        if (
            not force_evaluation
            and _is_source_only_performance(source_performance)
            and _is_target_report_only_performance(target_performance)
        ):
            skipped_folds.append(fold_number)
            if source_performance.get("passed") is not True:
                failed_folds.append(fold_number)
            print(
                f"[evaluation-only] fold {fold_number}: current evaluations "
                "already exist; skipped."
            )
            continue

        test_indices = np.asarray(record["test_indices"], dtype=np.int64)
        train_gene_indices = np.asarray(
            record.get(
                "train_indices",
                np.setdiff1d(np.arange(len(genes)), test_indices),
            ),
            dtype=np.int64,
        )
        current_fold = GeneFold(
            fold=fold_number,
            train_indices=train_gene_indices,
            test_indices=test_indices,
            train_genes=tuple(genes[index] for index in train_gene_indices),
            test_genes=tuple(genes[index] for index in test_indices),
        )
        fold_data = build_fold_data(prepared, current_fold, method="stdiff")
        source_train_indices = _validated_saved_indices(
            checkpoint.get("train_indices"),
            name="train_indices",
            upper_bound=len(fold_data.train_expression),
        )
        source_validation_indices = _validated_saved_indices(
            checkpoint.get("validation_indices"),
            name="validation_indices",
            upper_bound=len(fold_data.train_expression),
        )
        if torch.isin(source_train_indices, source_validation_indices).any():
            raise ValueError("Saved source train/validation indices overlap.")

        checkpoint_settings = checkpoint.get("training_settings")
        if not isinstance(checkpoint_settings, Mapping):
            checkpoint_settings = run_settings

        def setting(name: str) -> Any:
            return checkpoint_settings.get(
                name, run_settings.get(name, getattr(defaults, name))
            )

        validation_batch_size = int(setting("validation_batch_size"))
        validation_num_noise_samples = int(setting("validation_num_noise_samples"))
        validation_ode_steps = int(setting("validation_ode_steps"))
        max_validation_relative_mse = float(setting("max_validation_relative_mse"))
        min_validation_spcc = float(setting("min_validation_spcc"))
        model_seed = int(setting("model_seed"))
        num_workers = int(setting("num_workers"))

        train_expression = fold_data.train_expression[source_train_indices.numpy()]
        validation_expression = fold_data.train_expression[
            source_validation_indices.numpy()
        ]
        validation_condition = fold_data.train_condition[
            source_validation_indices.numpy()
        ]
        validation_dataloader = get_data_loader(
            validation_expression,
            validation_condition,
            batch_size=validation_batch_size,
            is_shuffle=False,
            seed=model_seed,
            num_workers=num_workers,
        )
        resolved_model_config = source_flow_model_config_from_checkpoint(checkpoint)
        model = build_source_flow_model(resolved_model_config).to(resolved_device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        checkpoint["model_config"] = resolved_model_config
        freeze_source_flow_model(model)
        validation_cfm_loss = evaluate_flow_matching_loss(
            model,
            validation_dataloader,
            device=resolved_device,
            seed=model_seed + 10_000 + fold_number,
        )
        print(f"[evaluation-only] fold {fold_number}: evaluating saved weights.")
        source_performance = evaluate_flow_imputation(
            model,
            torch.as_tensor(validation_expression),
            torch.as_tensor(validation_condition),
            target_gene_indices=test_indices,
            baseline_mean=torch.as_tensor(train_expression.mean(axis=0)),
            genes=genes,
            spot_names=np.asarray(prepared.adata_seq.obs_names)[
                source_validation_indices.numpy()
            ].tolist(),
            validation_cfm_loss=validation_cfm_loss,
            device=resolved_device,
            batch_size=validation_batch_size,
            num_noise_samples=validation_num_noise_samples,
            seed=model_seed + 20_000 + fold_number,
            ode_method="rk4",
            ode_steps=validation_ode_steps,
            ode_rtol=1e-5,
            ode_atol=1e-5,
            max_relative_mse=max_validation_relative_mse,
            min_median_spcc=min_validation_spcc,
            output_dir=root / "source_performance" / f"fold_{fold_number:02d}",
            metadata={
                "evaluation_domain": "source_validation",
                "evaluation_role": "checkpoint_filter",
                "baseline": "source_training_gene_mean",
                "target_missing_truth_used_for_filtering": False,
                "evaluation_only_revalidation": True,
            },
            apply_quality_gate=True,
        ).metrics

        saved_target_indices = checkpoint.get(
            "target_transfer_spot_indices",
            checkpoint.get("performance_spot_indices"),
        )
        if saved_target_indices is None:
            target_generator = torch.Generator(device="cpu").manual_seed(
                model_seed + 30_000 + fold_number
            )
            target_transfer_indices = torch.randperm(
                len(fold_data.spatial_expression), generator=target_generator
            )[: int(setting("validation_max_samples"))]
        else:
            target_transfer_indices = _validated_saved_indices(
                saved_target_indices,
                name="target_transfer_spot_indices",
                upper_bound=len(fold_data.spatial_expression),
            )
        target_performance = evaluate_flow_imputation(
            model,
            torch.as_tensor(
                fold_data.spatial_expression[target_transfer_indices.numpy()]
            ),
            torch.as_tensor(
                fold_data.spatial_condition[target_transfer_indices.numpy()]
            ),
            target_gene_indices=test_indices,
            baseline_mean=torch.as_tensor(train_expression.mean(axis=0)),
            genes=genes,
            spot_names=np.asarray(prepared.adata_spatial.obs_names)[
                target_transfer_indices.numpy()
            ].tolist(),
            validation_cfm_loss=validation_cfm_loss,
            device=resolved_device,
            batch_size=validation_batch_size,
            num_noise_samples=validation_num_noise_samples,
            seed=model_seed + 40_000 + fold_number,
            ode_method="rk4",
            ode_steps=validation_ode_steps,
            ode_rtol=1e-5,
            ode_atol=1e-5,
            max_relative_mse=max_validation_relative_mse,
            min_median_spcc=min_validation_spcc,
            output_dir=(
                root / "target_transfer_performance" / f"fold_{fold_number:02d}"
            ),
            metadata={
                "evaluation_domain": "target_spatial",
                "evaluation_role": "report_only",
                "baseline": "source_training_gene_mean",
                "used_for_checkpoint_filtering": False,
                "used_for_model_selection": False,
                "used_as_adaptation_signal": False,
                "missing_gene_truth_usage": "offline_evaluation_only",
                "evaluation_only_revalidation": True,
            },
            apply_quality_gate=False,
        ).metrics

        legacy_performance = checkpoint.get("performance")
        if (
            isinstance(legacy_performance, Mapping)
            and not _is_source_only_performance(legacy_performance)
            and "legacy_target_filter_performance" not in checkpoint
        ):
            checkpoint["legacy_target_filter_performance"] = dict(legacy_performance)
        checkpoint["source_performance"] = source_performance
        checkpoint["target_transfer_performance"] = target_performance
        checkpoint["target_transfer_spot_indices"] = target_transfer_indices
        checkpoint["performance"] = source_performance
        checkpoint["training_status"] = "complete"
        checkpoint["evaluation_only_revalidation"] = {
            "weights_retrained": False,
            "optimizer_steps": 0,
            "source_split": "saved_train_validation_indices",
            "legacy_target_performance_preserved": (
                "legacy_target_filter_performance" in checkpoint
            ),
        }
        _atomic_torch_save(checkpoint, checkpoint_path)
        processed_folds.append(fold_number)
        if source_performance.get("passed") is not True:
            failed_folds.append(fold_number)

    all_source_performance: dict[str, dict[str, object]] = {}
    all_target_performance: dict[str, dict[str, object]] = {}
    remaining_unvalidated_folds: list[int] = []
    for record in fold_records:
        fold_number = int(record["fold"])
        checkpoint_path = checkpoint_dir / f"fold_{fold_number:02d}.pt"
        if not checkpoint_path.is_file():
            remaining_unvalidated_folds.append(fold_number)
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source_performance = checkpoint.get("source_performance")
        target_performance = checkpoint.get("target_transfer_performance")
        if _is_source_only_performance(source_performance):
            all_source_performance[f"fold_{fold_number:02d}"] = dict(source_performance)
        else:
            remaining_unvalidated_folds.append(fold_number)
        if _is_target_report_only_performance(target_performance):
            all_target_performance[f"fold_{fold_number:02d}"] = dict(target_performance)

    save_json(all_source_performance, root / "source_flow_performance.json")
    save_json(
        all_target_performance,
        root / "target_transfer_performance.json",
    )
    algorithm_notes = experiment.setdefault("algorithm_notes", {})
    algorithm_notes["checkpoint_filtering_domain"] = "source_validation"
    algorithm_notes["target_transfer_evaluation"] = "report_only"
    algorithm_notes["evaluation_only_revalidation_supported"] = True
    save_json(experiment, experiment_path)
    save_json(
        {
            "mode": "evaluation_only",
            "weights_retrained": False,
            "optimizer_steps": 0,
            "processed_folds": processed_folds,
            "skipped_folds": skipped_folds,
            "failed_folds": sorted(set(failed_folds)),
            "remaining_unvalidated_folds": remaining_unvalidated_folds,
            "force_evaluation": force_evaluation,
        },
        root / "source_flow_revalidation.json",
    )
    if failed_folds and fail_on_performance:
        raise RuntimeError(
            "Source-only checkpoint performance gate failed for folds "
            f"{sorted(set(failed_folds))}; weights were not retrained."
        )
    return root


def train_source_flow_experiment(
    *,
    output_dir: str,
    dataset: str = "dataset5",
    data_root: str = "/Dataset/yezhong/ai4bio/stDiff_dataset",
    sc_path: Optional[str] = None,
    spatial_path: Optional[str] = None,
    device: Optional[str] = None,
    settings: Optional[SourceFlowTrainSettings] = None,
    fold: Optional[int] = None,
    start_fold: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    allow_existing: bool = False,
) -> Path:
    """Train one or all source-flow folds without using ST targets in the loss."""

    try:
        from .Dataloader import (
            GENE_LOCAL_PREPROCESSING,
            build_fold_data,
            build_gene_folds,
            fold_records,
            get_data_loader,
            load_paired_h5ad,
            preprocess_pair,
        )
        from .utils import (
            ensure_output_dir,
            environment_snapshot,
            load_json,
            resolve_device,
            save_json,
            set_seed,
            sha256_file,
        )
    except ImportError:
        from Dataloader import (
            GENE_LOCAL_PREPROCESSING,
            build_fold_data,
            build_gene_folds,
            fold_records,
            get_data_loader,
            load_paired_h5ad,
            preprocess_pair,
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

    configuration = settings or SourceFlowTrainSettings()
    configuration.validate()
    resolved_device = resolve_device(device)
    resolved_amp_dtype = resolve_amp_dtype(resolved_device, configuration.amp_mode)
    compiled_backward_autocast = _configure_compiled_backward_autocast(
        compile_model=configuration.compile_model,
        amp_dtype=resolved_amp_dtype,
    )
    numerical_contract = {
        "version": 1,
        "requested_amp_mode": configuration.amp_mode,
        "resolved_amp_dtype": (
            None
            if resolved_amp_dtype is None
            else str(resolved_amp_dtype).removeprefix("torch.")
        ),
        "torch_compile": configuration.compile_model,
        "compiled_backward_autocast": compiled_backward_autocast,
        "loss_finite_check": True,
        "gradient_finite_check": True,
        "parameter_finite_check": True,
        "validation_finite_check": True,
    }
    destination = ensure_output_dir(output_dir, allow_existing=allow_existing)
    checkpoint_dir = destination / "checkpoints"
    history_dir = destination / "training_history"
    sc_file, spatial_file, dataset_name = _resolve_input_paths(
        dataset, data_root, sc_path, spatial_path
    )
    adata_seq, adata_spatial = load_paired_h5ad(sc_file, spatial_file)
    prepared = preprocess_pair(
        adata_seq,
        adata_spatial,
        method="stdiff",
        noise_std=configuration.noise_std,
        leiden_random_state=configuration.split_seed,
        preprocessing=GENE_LOCAL_PREPROCESSING,
    )
    folds = build_gene_folds(
        prepared.genes,
        n_splits=configuration.n_splits,
        random_state=configuration.split_seed,
    )
    if fold is not None and start_fold is not None:
        raise ValueError("fold and start_fold are mutually exclusive.")
    if fold is not None:
        selected_folds = [item for item in folds if item.fold == fold]
        if not selected_folds:
            raise ValueError(f"fold must be in [1,{len(folds)}], got {fold}.")
    elif start_fold is not None:
        if not 1 <= start_fold <= len(folds):
            raise ValueError(
                f"start_fold must be in [1,{len(folds)}], got {start_fold}."
            )
        selected_folds = [item for item in folds if item.fold >= start_fold]
    else:
        selected_folds = folds

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

    model_config = {
        "architecture": GENE_ATTENTION_ARCHITECTURE,
        "input_size": int(prepared.seq_array.shape[1]),
        "condition_size": int(prepared.seq_array.shape[1]),
        "hidden_size": configuration.hidden_size,
        "depth": configuration.depth,
        "num_heads": configuration.num_heads,
        "classes": 6,
        "mlp_ratio": configuration.mlp_ratio,
        "time_embedding_scale": configuration.time_embedding_scale,
    }
    experiment = {
        "method": "source_flow",
        "preprocessing": GENE_LOCAL_PREPROCESSING,
        "dataset": dataset_name,
        "output_dir": str(destination),
        "device": str(resolved_device),
        "settings": asdict(configuration),
        "model": model_config,
        "environment": environment_snapshot(),
        "numerical_training_contract": numerical_contract,
        "algorithm_notes": {
            "source_distribution": "standard_normal",
            "probability_path": "linear",
            "path_schedule": "alpha_t=t,sigma_t=1-t",
            "objective": "conditional_flow_matching_velocity_mse",
            "source_model_architecture": GENE_ATTENTION_ARCHITECTURE,
            "attention_axis": "genes_within_spot",
            "row_separable": True,
            "training_samples": "source_scRNA_only",
            "condition": "existing_fold_masked_source_expression",
            "spatial_targets_used_for_training": False,
            "checkpoint_filtering_domain": "source_validation",
            "target_transfer_evaluation": "report_only",
            "numerical_training_contract": numerical_contract,
            "checkpoint_selection": (
                "best_source_validation_loss"
                if configuration.early_stopping_patience
                else "last_epoch"
            ),
        },
    }
    existing_experiment_path = destination / "experiment.json"
    existing_data_info_path = destination / "data_info.json"
    if allow_existing and existing_experiment_path.is_file():
        existing_experiment = load_json(existing_experiment_path)
        existing_data_info = (
            load_json(existing_data_info_path)
            if existing_data_info_path.is_file()
            else None
        )
        compatible = bool(
            isinstance(existing_experiment, Mapping)
            and isinstance(existing_data_info, Mapping)
            and existing_experiment.get("method") == "source_flow"
            and existing_experiment.get("preprocessing") == GENE_LOCAL_PREPROCESSING
            and existing_experiment.get("settings") == asdict(configuration)
            and existing_experiment.get("model") == model_config
            and existing_experiment.get("numerical_training_contract")
            == numerical_contract
            and existing_data_info.get("sc_sha256") == data_info["sc_sha256"]
            and existing_data_info.get("spatial_sha256") == data_info["spatial_sha256"]
            and existing_data_info.get("genes") == data_info["genes"]
        )
        if not compatible:
            raise ValueError(
                "--allow-existing cannot mix this run with artifacts created "
                "under different data, model, training, or numerical settings. "
                "Use a fresh --output-dir."
            )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    save_json(data_info, destination / "data_info.json")
    save_json(fold_records(folds), destination / "gene_folds.json")
    save_json(experiment, destination / "experiment.json")

    all_history: list[dict[str, float]] = []
    all_source_performance: dict[str, dict[str, object]] = {}
    all_target_transfer_performance: dict[str, dict[str, object]] = {}
    failed_folds = []
    for current_fold in selected_folds:
        set_seed(configuration.model_seed, deterministic=configuration.deterministic)
        fold_data = build_fold_data(prepared, current_fold, method="stdiff")
        split_generator = torch.Generator(device="cpu").manual_seed(
            configuration.model_seed + current_fold.fold
        )
        train_indices, validation_indices = _split_source_cells(
            len(fold_data.train_expression),
            configuration.validation_fraction,
            generator=split_generator,
        )
        if max_train_samples is not None:
            if max_train_samples < 1:
                raise ValueError("max_train_samples must be positive.")
            train_indices = train_indices[:max_train_samples]
        validation_indices = validation_indices[: configuration.validation_max_samples]
        target_transfer_generator = torch.Generator(device="cpu").manual_seed(
            configuration.model_seed + 30_000 + current_fold.fold
        )
        target_transfer_indices = torch.randperm(
            len(fold_data.spatial_expression), generator=target_transfer_generator
        )[: configuration.validation_max_samples]
        train_expression = fold_data.train_expression[train_indices.numpy()]
        train_condition = fold_data.train_condition[train_indices.numpy()]
        validation_expression = fold_data.train_expression[validation_indices.numpy()]
        validation_condition = fold_data.train_condition[validation_indices.numpy()]
        dataloader = get_data_loader(
            train_expression,
            train_condition,
            batch_size=configuration.batch_size,
            is_shuffle=True,
            seed=configuration.model_seed,
            num_workers=configuration.num_workers,
        )
        validation_dataloader = get_data_loader(
            validation_expression,
            validation_condition,
            batch_size=configuration.validation_batch_size,
            is_shuffle=False,
            seed=configuration.model_seed,
            num_workers=configuration.num_workers,
        )
        model = build_source_flow_model(model_config)
        fold_history_path = history_dir / f"fold_{current_fold.fold:02d}.json"

        def persist_fold_history(
            records: Sequence[Mapping[str, float]],
            *,
            fold_number: int = current_fold.fold,
            path: Path = fold_history_path,
        ) -> None:
            payload = [
                {"fold": float(fold_number), **dict(record)} for record in records
            ]
            temporary = path.with_name(f"{path.name}.tmp")
            save_json(payload, temporary)
            temporary.replace(path)

        history, optimizer = train_one_source_flow_fold(
            model,
            dataloader,
            learning_rate=configuration.learning_rate,
            epochs=configuration.epochs,
            device=resolved_device,
            show_progress=configuration.show_progress,
            validation_dataloader=validation_dataloader,
            validation_seed=configuration.model_seed + 10_000 + current_fold.fold,
            amp_mode=configuration.amp_mode,
            compile_model=configuration.compile_model,
            early_stopping_patience=configuration.early_stopping_patience,
            early_stopping_min_delta=configuration.early_stopping_min_delta,
            history_callback=persist_fold_history,
        )
        for record in history:
            all_history.append({"fold": float(current_fold.fold), **record})
        selected_records = [
            record for record in history if record.get("selected_checkpoint") == 1.0
        ]
        if len(selected_records) != 1:
            raise RuntimeError("Training history does not identify one selected epoch.")
        selected_record = selected_records[0]
        selected_epoch = int(selected_record["epoch"])
        stopped_epoch = int(history[-1]["epoch"])
        checkpoint_path = checkpoint_dir / f"fold_{current_fold.fold:02d}.pt"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "preprocessing": GENE_LOCAL_PREPROCESSING,
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_state_reset_after_best_restore": bool(
                configuration.early_stopping_patience
                and selected_epoch != stopped_epoch
            ),
            "method": "source_flow",
            "flow_matching_contract": {
                "version": 1,
                "architecture": GENE_ATTENTION_ARCHITECTURE,
                "row_separable": True,
                "attention_axis": "genes_within_spot",
                "base_distribution": "standard_normal",
                "probability_path": "linear",
                "path_schedule": "alpha_t=t,sigma_t=1-t",
                "velocity_target": "z-x0",
                "time_distribution": "uniform_[0,1)",
            },
            "numerical_training_contract": numerical_contract,
            "training_status": "evaluation_pending",
            "fold": current_fold.fold,
            "test_indices": current_fold.test_indices,
            "test_genes": list(current_fold.test_genes),
            "genes": prepared.genes.tolist(),
            "mask": fold_data.mask,
            "model_config": model_config,
            "training_settings": asdict(configuration),
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "target_transfer_spot_indices": target_transfer_indices,
            "epoch": selected_epoch,
            "training_stopped_epoch": stopped_epoch,
            "checkpoint_selection": (
                "best_source_validation_loss"
                if configuration.early_stopping_patience
                else "last_epoch"
            ),
        }
        # Persist finite trained weights before ODE/metric evaluation.  If an
        # evaluation process fails, --evaluation-only can finish this fold
        # without repeating training.
        _atomic_torch_save(checkpoint, checkpoint_path)
        source_performance = evaluate_flow_imputation(
            model,
            torch.as_tensor(validation_expression),
            torch.as_tensor(validation_condition),
            target_gene_indices=current_fold.test_indices,
            baseline_mean=torch.as_tensor(train_expression.mean(axis=0)),
            genes=prepared.genes.tolist(),
            spot_names=np.asarray(prepared.adata_seq.obs_names)[
                validation_indices.numpy()
            ].tolist(),
            validation_cfm_loss=selected_record["validation_flow_matching_loss"],
            device=resolved_device,
            batch_size=configuration.validation_batch_size,
            num_noise_samples=configuration.validation_num_noise_samples,
            seed=configuration.model_seed + 20_000 + current_fold.fold,
            ode_method="rk4",
            ode_steps=configuration.validation_ode_steps,
            ode_rtol=1e-5,
            ode_atol=1e-5,
            max_relative_mse=configuration.max_validation_relative_mse,
            min_median_spcc=configuration.min_validation_spcc,
            output_dir=(
                destination / "source_performance" / f"fold_{current_fold.fold:02d}"
            ),
            metadata={
                "evaluation_domain": "source_validation",
                "evaluation_role": "checkpoint_filter",
                "baseline": "source_training_gene_mean",
                "target_missing_truth_used_for_filtering": False,
            },
            apply_quality_gate=True,
        ).metrics
        target_transfer_performance = evaluate_flow_imputation(
            model,
            torch.as_tensor(
                fold_data.spatial_expression[target_transfer_indices.numpy()]
            ),
            torch.as_tensor(
                fold_data.spatial_condition[target_transfer_indices.numpy()]
            ),
            target_gene_indices=current_fold.test_indices,
            baseline_mean=torch.as_tensor(train_expression.mean(axis=0)),
            genes=prepared.genes.tolist(),
            spot_names=np.asarray(prepared.adata_spatial.obs_names)[
                target_transfer_indices.numpy()
            ].tolist(),
            validation_cfm_loss=selected_record["validation_flow_matching_loss"],
            device=resolved_device,
            batch_size=configuration.validation_batch_size,
            num_noise_samples=configuration.validation_num_noise_samples,
            seed=configuration.model_seed + 40_000 + current_fold.fold,
            ode_method="rk4",
            ode_steps=configuration.validation_ode_steps,
            ode_rtol=1e-5,
            ode_atol=1e-5,
            max_relative_mse=configuration.max_validation_relative_mse,
            min_median_spcc=configuration.min_validation_spcc,
            output_dir=(
                destination
                / "target_transfer_performance"
                / f"fold_{current_fold.fold:02d}"
            ),
            metadata={
                "evaluation_domain": "target_spatial",
                "evaluation_role": "report_only",
                "baseline": "source_training_gene_mean",
                "used_for_checkpoint_filtering": False,
                "used_for_model_selection": False,
                "used_as_adaptation_signal": False,
                "missing_gene_truth_usage": "offline_evaluation_only",
            },
            apply_quality_gate=False,
        ).metrics
        all_source_performance[f"fold_{current_fold.fold:02d}"] = source_performance
        all_target_transfer_performance[
            f"fold_{current_fold.fold:02d}"
        ] = target_transfer_performance
        if not source_performance["passed"]:
            failed_folds.append(current_fold.fold)
        checkpoint.update(
            {
                "training_status": "complete",
                "source_performance": source_performance,
                "target_transfer_performance": target_transfer_performance,
                # Backward-compatible alias.  Unlike legacy checkpoints, this is
                # source-validation performance and is safe for filtering.
                "performance": source_performance,
            }
        )
        _atomic_torch_save(checkpoint, checkpoint_path)

    history_by_fold: dict[int, list[dict[str, float]]] = {}
    aggregate_history_path = destination / "train_history.csv"
    try:
        import pandas as pd

        if allow_existing and aggregate_history_path.is_file():
            existing_frame = pd.read_csv(aggregate_history_path)
            if "fold" in existing_frame:
                for fold_number, frame in existing_frame.groupby("fold"):
                    records = frame.to_dict(orient="records")
                    if all(
                        np.isfinite(float(record["flow_matching_loss"]))
                        and np.isfinite(float(record["validation_flow_matching_loss"]))
                        for record in records
                    ):
                        history_by_fold[int(fold_number)] = records
        for history_path in sorted(history_dir.glob("fold_*.json")):
            records = load_json(history_path)
            if not isinstance(records, list) or not records:
                raise ValueError(f"Invalid fold history: {history_path}")
            fold_number = int(float(records[0]["fold"]))
            history_by_fold[fold_number] = records
        all_history = [
            record
            for fold_number in sorted(history_by_fold)
            for record in history_by_fold[fold_number]
        ]
        pd.DataFrame(all_history).to_csv(aggregate_history_path, index=False)
    except ImportError:
        np.savetxt(
            aggregate_history_path,
            np.asarray(
                [
                    [
                        record["fold"],
                        record["epoch"],
                        record["flow_matching_loss"],
                        record["validation_flow_matching_loss"],
                        record["selected_checkpoint"],
                    ]
                    for record in all_history
                ]
            ),
            delimiter=",",
            header=(
                "fold,epoch,flow_matching_loss,"
                "validation_flow_matching_loss,selected_checkpoint"
            ),
            comments="",
        )

    # Rebuild summaries from every complete checkpoint so a targeted rerun
    # (for example, --fold 2 --allow-existing) does not erase fold 1 metadata.
    all_source_performance = {}
    all_target_transfer_performance = {}
    failed_folds = []
    for fold_record in folds:
        fold_number = int(fold_record.fold)
        saved_checkpoint_path = checkpoint_dir / f"fold_{fold_number:02d}.pt"
        if not saved_checkpoint_path.is_file():
            continue
        saved_checkpoint = torch.load(
            saved_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if saved_checkpoint.get("training_status", "complete") != "complete":
            continue
        source_record = saved_checkpoint.get("source_performance")
        target_record = saved_checkpoint.get("target_transfer_performance")
        key = f"fold_{fold_number:02d}"
        if _is_source_only_performance(source_record):
            all_source_performance[key] = dict(source_record)
            if source_record.get("passed") is not True:
                failed_folds.append(fold_number)
        if _is_target_report_only_performance(target_record):
            all_target_transfer_performance[key] = dict(target_record)
    save_json(
        all_source_performance,
        destination / "source_flow_performance.json",
    )
    save_json(
        all_target_transfer_performance,
        destination / "target_transfer_performance.json",
    )
    if failed_folds and configuration.fail_on_performance:
        raise RuntimeError(
            "Source-only checkpoint performance gate failed for folds "
            f"{failed_folds}; downstream loading will reject these checkpoints."
        )
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train row-separable per-spot gene-attention source "
            "flow-matching models."
        )
    )
    parser.add_argument("--dataset", default="dataset5")
    parser.add_argument("--data-root", default="/Dataset/yezhong/ai4bio/stDiff_dataset")
    parser.add_argument("--sc-path", default=None)
    parser.add_argument("--spatial-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--noise-std", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=1202)
    parser.add_argument("--time-embedding-scale", type=float, default=1000.0)
    parser.add_argument("--num-workers", type=int, default=0)
    fold_selection = parser.add_mutually_exclusive_group()
    fold_selection.add_argument("--fold", type=int, default=None)
    fold_selection.add_argument(
        "--start-fold",
        type=int,
        default=None,
        help="Train this fold and every later fold in the same run.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--validation-max-samples", type=int, default=512)
    parser.add_argument("--validation-num-noise-samples", type=int, default=4)
    parser.add_argument("--validation-ode-steps", type=int, default=20)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "Stop after this many epochs without a sufficient source-validation "
            "CFM-loss improvement and restore the best epoch; 0 disables it."
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum source-validation CFM-loss decrease counted as improvement.",
    )
    parser.add_argument("--max-validation-relative-mse", type=float, default=1.0)
    parser.add_argument("--min-validation-spcc", type=float, default=0.0)
    parser.add_argument("--allow-failed-performance", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help=(
            "Load checkpoints from --output-dir and add source-only plus "
            "target report-only evaluations without retraining."
        ),
    )
    parser.add_argument(
        "--force-evaluation",
        action="store_true",
        help="Recompute evaluation-only metrics even when current records exist.",
    )
    parser.add_argument(
        "--amp",
        choices=("off", "auto", "bf16", "fp16"),
        default="auto",
        help="Mixed precision mode. auto prefers BF16 on supported CUDA devices.",
    )

    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile the training model with torch.compile.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.force_evaluation and not args.evaluation_only:
        raise ValueError("--force-evaluation requires --evaluation-only.")
    if args.evaluation_only and args.start_fold is not None:
        raise ValueError("--start-fold is only available during training.")
    if args.evaluation_only:
        output = reevaluate_source_flow_run(
            args.output_dir,
            device=args.device,
            fold=args.fold,
            sc_path=args.sc_path,
            spatial_path=args.spatial_path,
            force_evaluation=args.force_evaluation,
            fail_on_performance=not args.allow_failed_performance,
        )
        print(output)
        return
    settings = SourceFlowTrainSettings(
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        noise_std=args.noise_std,
        n_splits=args.n_splits,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        time_embedding_scale=args.time_embedding_scale,
        num_workers=args.num_workers,
        validation_fraction=args.validation_fraction,
        validation_batch_size=args.validation_batch_size,
        validation_max_samples=args.validation_max_samples,
        validation_num_noise_samples=args.validation_num_noise_samples,
        validation_ode_steps=args.validation_ode_steps,
        max_validation_relative_mse=args.max_validation_relative_mse,
        min_validation_spcc=args.min_validation_spcc,
        fail_on_performance=not args.allow_failed_performance,
        show_progress=not args.no_progress,
        deterministic=args.deterministic,
        amp_mode=args.amp,
        compile_model=args.torch_compile,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    output = train_source_flow_experiment(
        output_dir=args.output_dir,
        dataset=args.dataset,
        data_root=args.data_root,
        sc_path=args.sc_path,
        spatial_path=args.spatial_path,
        device=args.device,
        settings=settings,
        fold=args.fold,
        start_fold=args.start_fold,
        max_train_samples=args.max_train_samples,
        allow_existing=args.allow_existing,
    )
    print(output)


if __name__ == "__main__":
    main()


__all__ = [
    "SourceFlowTrainSettings",
    "train_one_source_flow_fold",
    "train_source_flow_experiment",
]
