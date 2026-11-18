"""Common technical metrics for stDiff, IMD-T, and external baselines."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
)

if TYPE_CHECKING:
    import anndata as ad


@dataclass
class MetricResult:
    gene_metrics: pd.DataFrame
    summary: pd.DataFrame
    clustering: pd.DataFrame | None = None


def fixed_rbf_mmd_squared(
    source: np.ndarray, target: np.ndarray, *, bandwidth: float
) -> float:
    """Biased MMD² for one preselected Gaussian bandwidth, including self pairs."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        source.ndim != 2 or target.ndim != 2
        or source.size == 0 or target.size == 0
        or source.shape[1] != target.shape[1]
    ):
        raise ValueError("MMD inputs must be nonempty matrices with the same feature dimension.")
    if not (np.isfinite(source).all() and np.isfinite(target).all()):
        raise ValueError("MMD inputs must be finite.")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("MMD bandwidth must be positive and finite.")

    def kernel_mean(left: np.ndarray, right: np.ndarray) -> float:
        total = 0.0
        # Kernel blocks bound memory while keeping every pair in the V-statistic.
        block_size = 512
        for start in range(0, len(left), block_size):
            for end in range(0, len(right), block_size):
                distances = cdist(left[start:start + block_size], right[end:end + block_size])
                with np.errstate(over="ignore"):
                    total += float(np.exp(-0.5 * (distances / bandwidth) ** 2).sum())
        return total / (len(left) * len(right))

    squared = kernel_mean(source, source) + kernel_mean(target, target) - 2 * kernel_mean(source, target)
    return max(0.0, squared)  # Roundoff can make an exact zero slightly negative.


_CALIBRATION_DELTA_COLUMNS = ("delta_mmd_squared", "delta_path_loss", "delta_mse")
_CALIBRATION_CORRELATION_KEYS = tuple(
    f"spearman_{column}_vs_delta_mse" for column in _CALIBRATION_DELTA_COLUMNS[:2]
)


def calibration_rank_correlations(delta_table: pd.DataFrame) -> dict:
    """Summarize the two descriptive Spearman coefficients across candidates."""

    if any(column not in delta_table for column in _CALIBRATION_DELTA_COLUMNS):
        raise ValueError(
            f"Expected one row per calibration candidate with columns {_CALIBRATION_DELTA_COLUMNS}."
        )
    deltas = delta_table[list(_CALIBRATION_DELTA_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(deltas).all():
        raise ValueError("Calibration score differences must be finite.")

    coefficients: dict[str, float | None] = {}
    undefined_reasons: dict[str, str] = {}
    for index, key in enumerate(_CALIBRATION_CORRELATION_KEYS):
        left, right = deltas[:, index], deltas[:, 2]
        if len(deltas) < 3:
            reason = "fewer_than_three_candidates"
        elif np.ptp(left) == 0 or np.ptp(right) == 0:
            reason = "constant_delta"
        else:
            coefficients[key] = float(stats.spearmanr(left, right)[0])
            continue
        coefficients[key] = None
        undefined_reasons[key] = reason
    return {"n_candidates": len(deltas), "undefined_reasons": undefined_reasons, **coefficients}


def _mean_defined_correlations(reports: Sequence[dict], count_key: str) -> dict:
    means = {}
    for key in _CALIBRATION_CORRELATION_KEYS:
        values = [report[key] for report in reports if report[key] is not None]
        means[key] = {"mean": float(np.mean(values)) if values else None, count_key: len(values)}
    return means


def summarize_cross_dataset_correlations(
    root: str | Path, datasets: Sequence[str], *, split_seed: int = 0
) -> dict:
    """Summarize within-fold Spearman coefficients from existing ranking outputs."""

    if not datasets or any(not dataset for dataset in datasets) or len(set(datasets)) != len(datasets):
        raise ValueError("Provide a nonempty list of distinct dataset names.")
    root = Path(root).expanduser()
    subsets = ("all_candidates", "nonnegative_delta_mmd_squared")
    comparable_settings = ("num_times", "fit_samples", "evaluation_samples", "rff_features", "ode_steps", "ridge")
    sampling_settings: dict = {}
    dataset_reports = []
    for dataset in datasets:
        pattern = re.compile(rf"{re.escape(dataset)}_seed{split_seed}_fold([1-9][0-9]*)")
        fold_directories = {}
        for path in root.iterdir():
            match = pattern.fullmatch(path.name)
            if match and path.is_dir():
                fold_directories[int(match[1])] = path
        if not fold_directories:
            raise FileNotFoundError(f"No ranking directories for {dataset}_seed{split_seed} in {root}.")

        folds = []
        for fold, path in sorted(fold_directories.items()):
            scores_path = path / "calibration_rank_scores.csv"
            table = pd.read_csv(scores_path)
            if (
                table.empty or "candidate" not in table
                or table["candidate"].isna().any() or table["candidate"].duplicated().any()
            ):
                raise ValueError(f"Expected one score row per distinct candidate in {scores_path}.")
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            settings = summary.get("settings", {})
            if settings.get("fold", fold) != fold:
                raise ValueError(f"Fold in {path / 'summary.json'} differs from its directory name.")
            test_spots, evaluated_spots = summary.get("test_spots"), summary.get("evaluated_spots")
            protocol = {key: settings[key] for key in comparable_settings if key in settings}
            if evaluated_spots is not None and test_spots is not None:
                protocol["spot_coverage"] = "all_test_spots" if evaluated_spots == test_spots else evaluated_spots
            if "target_seed_count" in summary:
                protocol["target_seed_count"] = summary["target_seed_count"]
            for key, value in protocol.items():
                if key in sampling_settings and sampling_settings[key] != value:
                    raise ValueError(
                        f"Incompatible ranking setting {key!r} in {path}: "
                        f"{value!r} differs from {sampling_settings[key]!r}."
                    )
                sampling_settings[key] = value

            all_candidates = calibration_rank_correlations(table)
            improved_mmd = table.loc[table["delta_mmd_squared"].astype(float) >= 0]
            folds.append({
                "fold": fold,
                "scores_path": str(scores_path),
                "n_test_spots": test_spots,
                "n_evaluated_spots": evaluated_spots,
                "n_target_seeds": summary.get("target_seed_count"),
                "evaluation_seed": settings.get("seed"),
                "sampling_settings": protocol,
                "all_candidates": all_candidates,
                "nonnegative_delta_mmd_squared": calibration_rank_correlations(improved_mmd),
            })
        dataset_reports.append({
            "dataset": dataset,
            "folds": folds,
            "fold_equal_mean": {
                subset: _mean_defined_correlations([fold[subset] for fold in folds], "n_defined_folds")
                for subset in subsets
            },
        })

    return {
        "split_seed": split_seed,
        "sampling_settings": sampling_settings,
        "datasets": dataset_reports,
        "dataset_equal_mean": {
            subset: _mean_defined_correlations([
                {key: item["mean"] for key, item in report["fold_equal_mean"][subset].items()}
                for report in dataset_reports
            ], "n_defined_datasets")
            for subset in subsets
        },
    }


def _as_frame(
    values: pd.DataFrame | np.ndarray,
    *,
    genes: Sequence[str] | None = None,
    index: Sequence[str] | None = None,
) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
    else:
        array = np.asarray(values)
        if array.ndim != 2:
            raise ValueError(f"Expected a two-dimensional matrix, got {array.shape}.")
        if genes is None:
            genes = [str(i) for i in range(array.shape[1])]
        frame = pd.DataFrame(array, columns=list(genes), index=index)
    frame.columns = frame.columns.astype(str)
    frame.index = frame.index.astype(str)
    if frame.columns.duplicated().any():
        frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]
    return frame.astype(float)


def align_prediction(
    ground_truth: pd.DataFrame,
    prediction: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require the same spot and gene identities, reordering only by labels."""

    if set(ground_truth.index) != set(prediction.index):
        raise ValueError("Prediction and ground truth do not contain the same spots.")
    if set(ground_truth.columns) != set(prediction.columns):
        raise ValueError("Prediction and ground truth do not contain the same genes.")
    prediction = prediction.loc[ground_truth.index, ground_truth.columns]
    if not np.isfinite(ground_truth.to_numpy()).all():
        raise ValueError("Ground truth contains NaN or Inf values.")
    if not np.isfinite(prediction.to_numpy()).all():
        raise ValueError("Prediction contains NaN or Inf values.")
    return ground_truth, prediction


def cal_ssim(im1: np.ndarray, im2: np.ndarray, data_range: float) -> float:
    """The scalar SSIM definition used by the stDiff prior-work script."""

    im1 = np.asarray(im1)
    im2 = np.asarray(im2)
    if im1.shape != im2.shape or im1.ndim != 2:
        raise ValueError("SSIM inputs must be equal-shape 2-D arrays.")
    mu1, mu2 = im1.mean(), im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    c3 = c2 / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        luminance = (2 * mu1 * mu2 + c1) / (mu1**2 + mu2**2 + c1)
        contrast = (2 * sigma1 * sigma2 + c2) / (sigma1**2 + sigma2**2 + c2)
        structure = (sigma12 + c3) / (sigma1 * sigma2 + c3)
    return float(luminance * contrast * structure)


def scale_max(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.divide(frame.max(axis=0), axis=1)


def scale_z_score(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        stats.zscore(frame.to_numpy(), axis=0),
        index=frame.index,
        columns=frame.columns,
    )


def scale_plus(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.divide(frame.sum(axis=0), axis=1)


def gene_metrics(
    ground_truth: pd.DataFrame | np.ndarray,
    prediction: pd.DataFrame | np.ndarray,
    *,
    genes: Sequence[str] | None = None,
    index: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Calculate SPCC, SSIM, RMSE, and JS once per gene."""

    raw = _as_frame(ground_truth, genes=genes, index=index)
    imputed = _as_frame(prediction, genes=genes, index=index)
    raw, imputed = align_prediction(raw, imputed)

    raw_max, imputed_max = scale_max(raw), scale_max(imputed)
    raw_z, imputed_z = scale_z_score(raw), scale_z_score(imputed)
    raw_plus, imputed_plus = scale_plus(raw), scale_plus(imputed)

    records: list[dict[str, float | str]] = []
    for gene in raw.columns:
        raw_column = raw[gene]
        imputed_column = imputed[gene]
        spcc, _ = stats.spearmanr(raw_column, imputed_column)

        raw_ssim = raw_max[gene].to_numpy().reshape(-1, 1)
        imputed_ssim = imputed_max[gene].to_numpy().reshape(-1, 1)
        data_range = float(max(raw_max[gene].max(), imputed_max[gene].max()))
        ssim = cal_ssim(raw_ssim, imputed_ssim, data_range)

        rmse = float(np.sqrt(((raw_z[gene] - imputed_z[gene]) ** 2).mean()))
        midpoint = (raw_plus[gene] + imputed_plus[gene]) / 2
        js = float(
            0.5 * stats.entropy(raw_plus[gene], midpoint)
            + 0.5 * stats.entropy(imputed_plus[gene], midpoint)
        )
        records.append(
            {
                "gene": gene,
                "SPCC": float(spcc),
                "SSIM": ssim,
                "RMSE": rmse,
                "JS": js,
            }
        )
    return pd.DataFrame(records).set_index("gene")


def clustering_metrics(
    adata: ad.AnnData,
    target: str,
    prediction: str,
) -> pd.DataFrame:
    target_values = adata.obs[target]
    predicted_values = adata.obs[prediction]
    return pd.DataFrame(
        [
            {
                "ARI": adjusted_rand_score(target_values, predicted_values),
                "AMI": adjusted_mutual_info_score(target_values, predicted_values),
                "Homo": homogeneity_score(target_values, predicted_values),
                "NMI": normalized_mutual_info_score(target_values, predicted_values),
            }
        ]
    )


def clustering_preservation(
    ground_truth: pd.DataFrame | np.ndarray,
    prediction: pd.DataFrame | np.ndarray,
    *,
    genes: Sequence[str] | None = None,
    index: Sequence[str] | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Reproduce the prior-work Leiden-on-GT versus Leiden-on-prediction check."""

    # Ordinary per-gene metrics do not require the optional clustering stack.
    import anndata as ad
    import scanpy as sc

    raw = _as_frame(ground_truth, genes=genes, index=index)
    imputed = _as_frame(prediction, genes=genes, index=index)
    raw, imputed = align_prediction(raw, imputed)
    raw_adata = ad.AnnData(
        X=raw.to_numpy(dtype=np.float32),
        obs=pd.DataFrame(index=raw.index),
        var=pd.DataFrame(index=raw.columns),
    )
    imputed_adata = ad.AnnData(
        X=imputed.to_numpy(dtype=np.float32),
        obs=pd.DataFrame(index=imputed.index),
        var=pd.DataFrame(index=imputed.columns),
    )
    for value in (raw_adata, imputed_adata):
        sc.tl.pca(value)
        sc.pp.neighbors(value, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(value, random_state=random_state)
    imputed_adata.obs["target"] = raw_adata.obs["leiden"].to_numpy()
    return clustering_metrics(imputed_adata, "target", "leiden")


def evaluate_prediction(
    prediction: pd.DataFrame | np.ndarray,
    ground_truth: pd.DataFrame | np.ndarray,
    *,
    genes: Sequence[str] | None = None,
    index: Sequence[str] | None = None,
    include_clustering: bool = True,
    random_state: int = 0,
    output_dir: str | Path | None = None,
) -> MetricResult:
    per_gene = gene_metrics(ground_truth, prediction, genes=genes, index=index)
    summary = per_gene.median(axis=0, skipna=True).to_frame().T
    summary.index = ["median"]
    clustering = None
    if include_clustering:
        clustering = clustering_preservation(
            ground_truth,
            prediction,
            genes=genes,
            index=index,
            random_state=random_state,
        )
        summary = pd.concat(
            [summary.reset_index(drop=True), clustering.reset_index(drop=True)],
            axis=1,
        )
        summary.index = ["median"]

    result = MetricResult(per_gene, summary, clustering)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        per_gene.to_csv(destination / "gene_metrics.csv", index=True)
        summary.to_csv(destination / "technical_metrics.csv", index=True)
        if clustering is not None:
            clustering.to_csv(destination / "clustering_metrics.csv", index=False)
    return result


__all__ = [
    "MetricResult",
    "align_prediction",
    "cal_ssim",
    "calibration_rank_correlations",
    "fixed_rbf_mmd_squared",
    "clustering_metrics",
    "clustering_preservation",
    "evaluate_prediction",
    "gene_metrics",
    "scale_max",
    "scale_plus",
    "scale_z_score",
]
