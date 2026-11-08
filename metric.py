"""Common technical metrics for stDiff, IMD-T, and external baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
import scipy.stats as stats
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
    "clustering_metrics",
    "clustering_preservation",
    "evaluate_prediction",
    "gene_metrics",
    "scale_max",
    "scale_plus",
    "scale_z_score",
]
