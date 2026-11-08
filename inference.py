"""Checkpoint-only inference for migrated stDiff and IMD-T."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from Dataloader import (
    LEGACY_PREPROCESSING,
    GeneFold,
    build_fold_data,
    get_data_loader,
    load_paired_h5ad,
    preprocess_pair,
)
from model import (
    NoiseScheduler,
    SpatialGraphSmoothing,
    build_model,
    checkpoint_attention_architecture,
    build_noise_scheduler,
    get_schedule_jump,
)
from utils import (
    ensure_output_dir,
    load_json,
    resolve_device,
    save_json,
    set_seed,
    sha256_file,
)


def model_sample(
    model: torch.nn.Module,
    *,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    total_sample: torch.Tensor,
    time: int,
    is_conditional: bool,
    condition_flag: bool = True,
) -> torch.Tensor:
    """Evaluate a full spatial sample in DataLoader-sized chunks."""

    outputs: list[torch.Tensor] = []
    offset = 0
    for _, condition in dataloader:
        condition = condition.to(device)
        timesteps = torch.full(
            (len(condition),), time, dtype=torch.long, device=device
        )
        sample_chunk = total_sample[offset : offset + len(condition)]
        if is_conditional:
            prediction = model(
                sample_chunk,
                timesteps,
                condition,
                condi_flag=condition_flag,
            )
        else:
            prediction = model(sample_chunk, timesteps, None)
        outputs.append(prediction)
        offset += len(condition)
    if offset != len(total_sample):
        raise RuntimeError(
            f"Inference DataLoader covered {offset} rows, expected {len(total_sample)}."
        )
    return torch.cat(outputs, dim=0)


def sample_stdiff(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    noise_scheduler: NoiseScheduler,
    *,
    mask: np.ndarray,
    ground_truth: np.ndarray,
    device: torch.device,
    num_steps: int,
    sample_shape: tuple[int, int],
    model_pred_type: str = "noise",
    show_progress: bool = True,
) -> np.ndarray:
    """Original stDiff stochastic reverse diffusion with hard mask refill."""

    model.eval()
    x_t = torch.randn(sample_shape, device=device)
    mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=device)
    gt_tensor = torch.as_tensor(
        ground_truth, dtype=torch.float32, device=device
    )
    x_t = x_t * (1 - mask_tensor) + gt_tensor * mask_tensor

    iterator = tqdm(
        list(range(num_steps))[::-1],
        disable=not show_progress,
        desc="stDiff sampling",
    )
    for time in iterator:
        with torch.no_grad():
            model_output = model_sample(
                model,
                device=device,
                dataloader=dataloader,
                total_sample=x_t,
                time=time,
                is_conditional=True,
            )
            x_t, _ = noise_scheduler.step(
                model_output,
                torch.tensor(time, dtype=torch.long, device=device),
                x_t,
                model_pred_type=model_pred_type,
            )
            x_t = x_t * (1.0 - mask_tensor) + mask_tensor * gt_tensor
    return x_t.detach().cpu().numpy()


def _extract_into_tensor(
    array: np.ndarray,
    timesteps: torch.Tensor,
    broadcast_shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    result = torch.from_numpy(array).to(device=timesteps.device)[timesteps].float()
    while len(result.shape) < len(broadcast_shape):
        result = result[..., None]
    return result.expand(broadcast_shape)


def repaint_inject(
    x_t: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    noise_scheduler: NoiseScheduler,
    timestep: int,
    *,
    use_cumulative_noise: bool = True,
) -> torch.Tensor:
    """Experimental IMD-T RePaint injection migrated from its sample.py."""

    if use_cumulative_noise:
        weighted_gt = noise_scheduler.get_gt_noised(ground_truth, timestep)
    else:
        timestep_tensor = torch.tensor([timestep], device=x_t.device)
        alpha_cumprod = _extract_into_tensor(
            noise_scheduler.alphas_cumprod.numpy(),
            timestep_tensor.cpu(),
            x_t.shape,
        ).to(x_t.device)
        noise = torch.randn_like(x_t)
        # Kept identical to the experimental prior-work expression.
        weighted_gt = (
            alpha_cumprod**0.5 * ground_truth
            + (1 - alpha_cumprod**0.5) ** 0.5
            * (1 - alpha_cumprod) ** 0.5
            * noise
        )
    return mask * weighted_gt + (1 - mask) * x_t


def sample_imdt(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    noise_scheduler: NoiseScheduler,
    *,
    mask: np.ndarray,
    ground_truth: np.ndarray,
    coordinates: np.ndarray,
    device: torch.device,
    num_steps: int,
    sample_shape: tuple[int, int],
    model_pred_type: str = "noise",
    use_repaint: bool = False,
    repaint_jump: int = 1,
    repaint_start: int = 100,
    use_cumulative_noise: bool = True,
    schedule_jump_params: Mapping[str, Any] | None = None,
    graph_k: int = 6,
    graph_lambda: float = 0.2,
    show_progress: bool = True,
    return_pre_smoothing: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """IMD-T default sampler plus its default-off experimental RePaint path.

    The default path deliberately preserves two upstream behaviours: the
    timestep-pair loop ends after the t=1 reverse step, and SG is calculated at
    every step but is not fed back into the next diffusion state.
    """

    if num_steps < 2:
        raise ValueError("IMD-T sampling requires at least two diffusion steps.")
    model.eval()
    x_t = torch.randn(sample_shape, device=device)
    mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=device)
    gt_tensor = torch.as_tensor(
        ground_truth, dtype=torch.float32, device=device
    )
    x_t = x_t * (1 - mask_tensor) + gt_tensor * mask_tensor

    if use_repaint:
        t_max = num_steps - 1
        alpha = noise_scheduler.alphas_cumprod[
            torch.tensor([t_max])
        ].reshape(-1, 1).to(device)
        gt_noised = alpha**0.5 * gt_tensor + (1 - alpha) ** 0.5 * torch.randn_like(
            x_t
        )
        x_t = mask_tensor * gt_noised + (1 - mask_tensor) * x_t

    # The upstream sample.py resets this cache immediately after initialisation.
    noise_scheduler.reset_gt_noises()
    if schedule_jump_params is None:
        times = list(range(num_steps))[::-1]
    else:
        times = get_schedule_jump(**dict(schedule_jump_params))
    time_pairs = list(zip(times[:-1], times[1:]))
    spatial_graph = SpatialGraphSmoothing(
        coordinates, K=graph_k, device=device
    )

    pre_smoothing: torch.Tensor | None = None
    post_smoothing: torch.Tensor | None = None
    predicted_x0: torch.Tensor | None = None
    iterator = tqdm(
        time_pairs,
        disable=not show_progress,
        desc="IMD-T sampling",
    )
    for previous_time, current_time in iterator:
        timestep = torch.tensor(
            [previous_time], dtype=torch.long, device=device
        )
        forward_jump = current_time > previous_time
        with torch.no_grad():
            if forward_jump and use_repaint and previous_time < repaint_start:
                for _ in range(repaint_jump):
                    x_t = repaint_inject(
                        x_t,
                        gt_tensor,
                        mask_tensor,
                        noise_scheduler,
                        previous_time,
                        use_cumulative_noise=use_cumulative_noise,
                    )
                    model_output = model_sample(
                        model,
                        device=device,
                        dataloader=dataloader,
                        total_sample=x_t,
                        time=previous_time,
                        is_conditional=True,
                    )
                    x_t, predicted_x0 = noise_scheduler.step(
                        model_output,
                        timestep,
                        x_t,
                        model_pred_type=model_pred_type,
                    )
                x_t = noise_scheduler.undo(
                    x_t.clone(), x_t, predicted_x0, timestep
                )
            elif forward_jump:
                x_t = noise_scheduler.undo(x_t.clone(), x_t, None, timestep)
            else:
                if use_repaint and previous_time < repaint_start:
                    x_t = repaint_inject(
                        x_t,
                        gt_tensor,
                        mask_tensor,
                        noise_scheduler,
                        previous_time,
                        use_cumulative_noise=use_cumulative_noise,
                    )
                model_output = model_sample(
                    model,
                    device=device,
                    dataloader=dataloader,
                    total_sample=x_t,
                    time=previous_time,
                    is_conditional=True,
                )
                x_t, predicted_x0 = noise_scheduler.step(
                    model_output,
                    timestep,
                    x_t,
                    model_pred_type=model_pred_type,
                )
                if not use_repaint:
                    x_t = x_t * (1.0 - mask_tensor) + mask_tensor * gt_tensor
                    pre_smoothing = x_t
                    smoothed = (1 - graph_lambda) * x_t + graph_lambda * spatial_graph.SG(
                        x_t
                    )
                    post_smoothing = (
                        smoothed * (1.0 - mask_tensor) + mask_tensor * gt_tensor
                    )
        if show_progress:
            iterator.set_description_str(
                f"IMD-T t={previous_time}->{current_time}"
            )

    if use_repaint:
        # The upstream branch does not define x_scid; returning x_t makes the
        # experimental branch inspectable without changing the default path.
        pre_smoothing = x_t
        post_smoothing = x_t
    if pre_smoothing is None or post_smoothing is None:
        raise RuntimeError("IMD-T sampler did not produce a final sample.")

    final = post_smoothing.detach().cpu().numpy()
    pre = pre_smoothing.detach().cpu().numpy()
    return (final, pre) if return_pre_smoothing else final


def _fold_from_record(record: Mapping[str, Any]) -> GeneFold:
    return GeneFold(
        fold=int(record["fold"]),
        train_indices=np.asarray(record["train_indices"], dtype=np.int64),
        test_indices=np.asarray(record["test_indices"], dtype=np.int64),
        train_genes=tuple(str(value) for value in record["train_genes"]),
        test_genes=tuple(str(value) for value in record["test_genes"]),
    )


def run_inference(
    *,
    checkpoint_dir: str | Path,
    output_dir: str | Path | None = None,
    sc_path: str | Path | None = None,
    spatial_path: str | Path | None = None,
    device: str | torch.device | None = None,
    inference_seed: int = 1202,
    batch_size: int | None = None,
    use_repaint: bool = False,
    repaint_params: Mapping[str, Any] | None = None,
    show_progress: bool = True,
    allow_existing: bool = False,
) -> pd.DataFrame:
    """Load a complete fold set and generate predictions without training."""

    checkpoints_path = Path(checkpoint_dir).expanduser().resolve()
    training_dir = checkpoints_path.parent
    experiment_path = training_dir / "experiment.json"
    fold_path = training_dir / "gene_folds.json"
    data_info_path = training_dir / "data_info.json"
    for required in (experiment_path, fold_path, data_info_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required training metadata not found: {required}")

    experiment = load_json(experiment_path)
    data_info = load_json(data_info_path)
    fold_records = load_json(fold_path)
    method = str(experiment["method"]).lower()
    settings = experiment["settings"]
    model_config = experiment["model"]
    resolved_device = resolve_device(device or experiment.get("device"))

    sc_file = Path(sc_path or data_info["sc_h5ad"])
    spatial_file = Path(spatial_path or data_info["spatial_h5ad"])
    if sha256_file(sc_file) != data_info["sc_sha256"]:
        raise ValueError("scRNA H5AD hash differs from the training input.")
    if sha256_file(spatial_file) != data_info["spatial_sha256"]:
        raise ValueError("Spatial H5AD hash differs from the training input.")

    adata_seq, adata_spatial = load_paired_h5ad(
        sc_file,
        spatial_file,
        require_spatial_coords=(method == "imdt"),
    )
    prepared = preprocess_pair(
        adata_seq,
        adata_spatial,
        method=method,
        noise_std=float(settings["noise_std"]),
        leiden_resolution=float(settings["leiden_resolution"]),
        leiden_random_state=int(settings["split_seed"]),
        preprocessing=experiment.get("preprocessing", LEGACY_PREPROCESSING),
    )
    if prepared.genes.tolist() != data_info["genes"]:
        raise ValueError("Gene order differs from training metadata.")

    folds = [_fold_from_record(record) for record in fold_records]
    checkpoint_files = sorted(checkpoints_path.glob("fold_*.pt"))
    if len(checkpoint_files) != len(folds):
        raise FileNotFoundError(
            f"Expected {len(folds)} fold checkpoints, found {len(checkpoint_files)}."
        )

    destination = ensure_output_dir(
        output_dir or (training_dir / "inference"),
        allow_existing=allow_existing,
    )
    set_seed(inference_seed)
    all_predictions = np.zeros_like(prepared.spatial_array, dtype=np.float32)
    all_pre_sg = (
        np.zeros_like(prepared.spatial_array, dtype=np.float32)
        if method == "imdt"
        else None
    )
    loader_batch_size = int(batch_size or settings["batch_size"])

    checkpoint_hashes: dict[str, str] = {}
    run_architecture = None
    for fold, checkpoint_file in zip(folds, checkpoint_files):
        checkpoint = torch.load(
            checkpoint_file,
            map_location=resolved_device,
            weights_only=False,
        )
        if int(checkpoint["fold"]) != fold.fold:
            raise ValueError(f"Fold mismatch in {checkpoint_file}.")
        if checkpoint["method"] != method:
            raise ValueError(f"Method mismatch in {checkpoint_file}.")
        if list(checkpoint["genes"]) != prepared.genes.tolist():
            raise ValueError(f"Gene identity mismatch in {checkpoint_file}.")
        architecture = checkpoint_attention_architecture(method, model_config, checkpoint)
        if run_architecture is not None and run_architecture != architecture:
            raise ValueError(f"Mixed attention architectures in {checkpoints_path}.")
        run_architecture = architecture

        fold_data = build_fold_data(prepared, fold, method=method)
        inference_loader = get_data_loader(
            fold_data.spatial_ground_truth,
            fold_data.spatial_condition,
            batch_size=loader_batch_size,
            is_shuffle=False,
            seed=inference_seed,
            num_workers=int(settings["num_workers"]),
        )
        model = build_model(
            method,
            input_size=int(model_config["input_size"]),
            hidden_size=int(model_config["hidden_size"]),
            depth=int(model_config["depth"]),
            num_heads=int(model_config["num_heads"]),
            classes=int(model_config["classes"]),
            mlp_ratio=float(model_config["mlp_ratio"]),
            architecture=architecture,
        ).to(resolved_device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        scheduler = build_noise_scheduler(
            method, int(settings["diffusion_steps"])
        )

        if method == "stdiff":
            fold_prediction = sample_stdiff(
                model,
                inference_loader,
                scheduler,
                mask=fold_data.mask,
                ground_truth=fold_data.spatial_ground_truth,
                device=resolved_device,
                num_steps=int(settings["diffusion_steps"]),
                sample_shape=prepared.spatial_array.shape,
                show_progress=show_progress,
            )
        else:
            params = dict(repaint_params or {})
            fold_prediction, fold_pre_sg = sample_imdt(
                model,
                inference_loader,
                scheduler,
                mask=fold_data.mask,
                ground_truth=fold_data.spatial_ground_truth,
                coordinates=prepared.coordinates,
                device=resolved_device,
                num_steps=int(settings["diffusion_steps"]),
                sample_shape=prepared.spatial_array.shape,
                use_repaint=use_repaint,
                repaint_jump=int(params.get("repaint_jump", 1)),
                repaint_start=int(params.get("repaint_start", 100)),
                use_cumulative_noise=bool(
                    params.get("use_cumulative_noise", True)
                ),
                schedule_jump_params=params.get("schedule_jump_params"),
                graph_k=int(params.get("graph_k", 6)),
                graph_lambda=float(params.get("graph_lambda", 0.2)),
                show_progress=show_progress,
                return_pre_smoothing=True,
            )
            all_pre_sg[:, fold.test_indices] = fold_pre_sg[:, fold.test_indices]

        all_predictions[:, fold.test_indices] = fold_prediction[:, fold.test_indices]
        checkpoint_hashes[checkpoint_file.name] = sha256_file(checkpoint_file)

    prediction = (all_predictions + 1.0) / 2.0
    prediction_frame = pd.DataFrame(
        prediction,
        index=adata_spatial.obs_names.copy(),
        columns=prepared.genes,
    )
    prediction_frame.to_csv(destination / "prediction.csv", index=True)
    prediction_adata = adata_spatial.copy()
    prediction_adata.X = prediction.astype(np.float32)
    prediction_adata.write_h5ad(destination / "prediction.h5ad")

    if all_pre_sg is not None:
        pre_sg = (all_pre_sg + 1.0) / 2.0
        pd.DataFrame(
            pre_sg,
            index=adata_spatial.obs_names.copy(),
            columns=prepared.genes,
        ).to_csv(destination / "prediction_pre_sg.csv", index=True)

    save_json(
        {
            "method": method,
            "architecture": run_architecture,
            "checkpoint_dir": str(checkpoints_path),
            "checkpoint_sha256": checkpoint_hashes,
            "inference_seed": inference_seed,
            "batch_size": loader_batch_size,
            "use_repaint": use_repaint,
            "repaint_params": dict(repaint_params or {}),
            "prediction_shape": list(prediction_frame.shape),
            "prediction_sha256": sha256_file(destination / "prediction.csv"),
        },
        destination / "inference.json",
    )
    return prediction_frame


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run checkpoint-only migrated stDiff/IMD-T inference"
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sc-path", default=None)
    parser.add_argument("--spatial-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1202)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--use-repaint", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_inference(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        sc_path=args.sc_path,
        spatial_path=args.spatial_path,
        device=args.device,
        inference_seed=args.seed,
        batch_size=args.batch_size,
        use_repaint=args.use_repaint,
        show_progress=not args.no_progress,
        allow_existing=args.allow_existing,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "model_sample",
    "repaint_inject",
    "run_inference",
    "sample_imdt",
    "sample_stdiff",
]
