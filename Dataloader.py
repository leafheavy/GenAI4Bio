"""Data loading and preprocessing shared by stDiff and IMD-T.

New runs use a gene-local transformation so hidden genes cannot alter visible
inputs. The prior per-spot transformation remains available for old checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from anndata import AnnData
from sklearn.model_selection import KFold
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_DATA_ROOT = Path("/Dataset/yezhong/ai4bio/stDiff_dataset")
GENE_LOCAL_PREPROCESSING = "gene_local_log1p_v1"
LEGACY_PREPROCESSING = "legacy_full_gene_v1"
DATASET_FILES = {
    "dataset1": (
        Path("sc/dataset1_seq_1670.h5ad"),
        Path("sp/dataset1_spatial_246_preprocess.h5ad"),
    ),
    "dataset2": (
        Path("sc/dataset2_seq_33.h5ad"),
        Path("sp/dataset2_spatial_33.h5ad"),
    ),
    "dataset3": (
        Path("sc/dataset3_seq_42.h5ad"),
        Path("sp/dataset3_spatial_42.h5ad"),
    ),
    "dataset4": (
        Path("sc/dataset4_seq_1000.h5ad"),
        Path("sp/dataset4_spatial_1000.h5ad"),
    ),
    "dataset5": (
        Path("sc/dataset5_seq_915.h5ad"),
        Path("sp/dataset5_spatial_915.h5ad"),
    ),
    "dataset6": (
        Path("sc/dataset6_seq_251.h5ad"),
        Path("sp/dataset6_spatial_251.h5ad"),
    ),
    "dataset7": (
        Path("sc/dataset7_seq_118.h5ad"),
        Path("sp/dataset7_spatial_118.h5ad"),
    ),
    "dataset8": (
        Path("sc/dataset8_seq_84.h5ad"),
        Path("sp/dataset8_spatial_84.h5ad"),
    ),
    "dataset9": (
        Path("sc/dataset9_seq_76.h5ad"),
        Path("sp/dataset9_spatial_76.h5ad"),
    ),
    "dataset10": (
        Path("sc/dataset10_seq_42.h5ad"),
        Path("sp/dataset10_spatial_42.h5ad"),
    ),
    "dataset11": (
        Path("sc/dataset11_seq_347.h5ad"),
        Path("sp/dataset11_spatial_347.h5ad"),
    ),
    "dataset12": (
        Path("sc/dataset12_seq_1000.h5ad"),
        Path("sp/dataset12_spatial_1000.h5ad"),
    ),
    "dataset13": (
        Path("sc/dataset13_seq_154.h5ad"),
        Path("sp/dataset13_spatial_154.h5ad"),
    ),
    "dataset14": (
        Path("sc/dataset14_seq_981.h5ad"),
        Path("sp/dataset14_spatial_981.h5ad"),
    ),
    "dataset15": (
        Path("sc/dataset15_seq_141.h5ad"),
        Path("sp/dataset15_spatial_141.h5ad"),
    ),
    "dataset16": (
        Path("sc/dataset16_seq_118.h5ad"),
        Path("sp/dataset16_spatial_118.h5ad"),
    ),
}


@dataclass(frozen=True)
class DatasetPaths:
    dataset: str
    sc_h5ad: Path
    spatial_h5ad: Path


@dataclass
class PreparedData:
    adata_seq: AnnData
    adata_spatial: AnnData
    seq_array: np.ndarray
    spatial_array: np.ndarray
    genes: np.ndarray
    seq_labels: np.ndarray | None = None
    spatial_labels: np.ndarray | None = None
    coordinates: np.ndarray | None = None


@dataclass(frozen=True)
class GeneFold:
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_genes: tuple[str, ...]
    test_genes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "train_indices": self.train_indices.tolist(),
            "test_indices": self.test_indices.tolist(),
            "train_genes": list(self.train_genes),
            "test_genes": list(self.test_genes),
        }


@dataclass
class FoldData:
    mask: np.ndarray
    train_expression: np.ndarray
    train_condition: np.ndarray
    spatial_expression: np.ndarray
    spatial_condition: np.ndarray
    spatial_ground_truth: np.ndarray
    test_indices: np.ndarray


def _normalise_dataset_name(dataset: str | int) -> str:
    value = str(dataset).strip().lower()
    if value.startswith("dataset"):
        key = value
    else:
        key = f"dataset{value}"
    if key not in DATASET_FILES:
        raise ValueError(f"Supported datasets are {sorted(DATASET_FILES)}, got {dataset!r}.")
    return key


def resolve_dataset_paths(
    dataset: str | int,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> DatasetPaths:
    """Resolve dataset2 or dataset5 under the requested server directory."""

    key = _normalise_dataset_name(dataset)
    sc_relative, spatial_relative = DATASET_FILES[key]
    root = Path(data_root).expanduser()
    return DatasetPaths(
        dataset=key,
        sc_h5ad=root / sc_relative,
        spatial_h5ad=root / spatial_relative,
    )


def _dense_float32(matrix: object) -> np.ndarray:
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def read_h5ad(path: str | Path) -> AnnData:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"H5AD file not found: {source}")
    adata = sc.read_h5ad(source)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    return adata


def load_paired_h5ad(
    sc_path: str | Path,
    spatial_path: str | Path,
    *,
    require_spatial_coords: bool = False,
) -> tuple[AnnData, AnnData]:
    adata_seq = read_h5ad(sc_path)
    adata_spatial = read_h5ad(spatial_path)
    validate_pair(
        adata_seq,
        adata_spatial,
        require_spatial_coords=require_spatial_coords,
    )
    return adata_seq, adata_spatial


def _finite_and_nonnegative(adata: AnnData, name: str) -> None:
    values = adata.X.data if sp.issparse(adata.X) else np.asarray(adata.X)
    if not np.isfinite(values).all():
        raise ValueError(f"{name}.X contains NaN or Inf values.")
    if np.any(values < 0):
        raise ValueError(f"{name}.X contains negative values before preprocessing.")


def get_spatial_coordinates(adata: AnnData) -> np.ndarray | None:
    if "spatial" in adata.obsm:
        coordinates = np.asarray(adata.obsm["spatial"])
        if coordinates.ndim == 2 and coordinates.shape[1] >= 2:
            return coordinates[:, :2].astype(np.float32)

    for x_key, y_key in (
        ("x_coord", "y_coord"),
        ("array_row", "array_col"),
        ("row", "col"),
        ("x", "y"),
        ("X", "Y"),
        ("xcoord", "ycoord"),
    ):
        if x_key in adata.obs and y_key in adata.obs:
            return adata.obs[[x_key, y_key]].to_numpy(dtype=np.float32)
    return None


def validate_pair(
    adata_seq: AnnData,
    adata_spatial: AnnData,
    *,
    require_spatial_coords: bool = False,
) -> None:
    """Validate the identity and numerical assumptions made by both methods."""

    for name, adata in (("scRNA", adata_seq), ("spatial", adata_spatial)):
        if adata.X.ndim != 2:
            raise ValueError(f"{name}.X must be two-dimensional, got {adata.X.ndim}D.")
        if not adata.var_names.is_unique:
            raise ValueError(f"{name} gene names are not unique.")
        if not adata.obs_names.is_unique:
            raise ValueError(f"{name} observation names are not unique.")
        _finite_and_nonnegative(adata, name)

    seq_genes = np.asarray(adata_seq.var_names, dtype=str)
    spatial_genes = np.asarray(adata_spatial.var_names, dtype=str)
    if not np.array_equal(seq_genes, spatial_genes):
        raise ValueError(
            "scRNA and spatial var_names must be identical and in the same order."
        )

    if require_spatial_coords:
        coordinates = get_spatial_coordinates(adata_spatial)
        if coordinates is None:
            raise ValueError("IMD-T requires spatial coordinates.")
        if not np.isfinite(coordinates).all():
            raise ValueError("Spatial coordinates contain NaN or Inf values.")
        if len(coordinates) < 7:
            raise ValueError("IMD-T K=6 graph smoothing requires at least 7 spots.")


def reindex(adata: AnnData, genes: Sequence[str], chunk_size: int = 20_000) -> AnnData:
    """Prior-work compatible AnnData reindexing helper."""

    genes = np.asarray(genes, dtype=str)
    present = np.isin(genes, np.asarray(adata.var_names, dtype=str))
    if present.all():
        return adata[:, genes].copy()

    new_x = sp.lil_matrix((adata.n_obs, len(genes)), dtype=np.float32)
    present_positions = np.flatnonzero(present)
    present_genes = genes[present]
    for start in range(0, adata.n_obs, chunk_size):
        stop = min(start + chunk_size, adata.n_obs)
        new_x[start:stop, present_positions] = adata[start:stop, present_genes].X
    return AnnData(
        new_x.tocsr(),
        obs=adata.obs.copy(),
        var={"var_names": genes},
    )


def data_augment(adata: AnnData, fixed: bool = True, noise_std: float = 10.0) -> AnnData:
    """Duplicate scRNA cells after adding the prior-work fixed/noisy offset."""

    augmented = adata.copy()
    if fixed:
        augmented.X = augmented.X + np.full(adata.shape, noise_std)
    else:
        augmented.X = augmented.X + np.abs(
            np.random.normal(0, noise_std, adata.shape)
        )

    # Keep the upstream concatenate semantics, including duplicated-cell suffixes.
    return adata.concatenate(augmented, join="outer")


def scale(adata: AnnData) -> AnnData:
    """Apply the original per-cell MaxAbs scaling implementation."""

    scaler = MaxAbsScaler()
    normalised = scaler.fit_transform(adata.X.T).T
    adata.X = normalised
    return adata


def scale_gene_local(adata: AnnData) -> AnnData:
    """Map each raw gene value to [0, 1) without reading other genes."""

    if sp.issparse(adata.X):
        values = adata.X.copy().astype(np.float32)
        values.data = np.log1p(values.data)
        values.data /= 1.0 + values.data
    else:
        values = np.log1p(np.asarray(adata.X, dtype=np.float32))
        values /= 1.0 + values
    adata.X = values
    return adata


def _cluster_leiden(
    adata: AnnData,
    *,
    resolution: float = 0.4,
    random_state: int = 0,
) -> np.ndarray:
    sc.pp.pca(adata, n_comps=min(10, adata.n_vars))
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state)
    return adata.obs["leiden"].cat.codes.to_numpy(dtype=np.int64)


def preprocess_pair(
    adata_seq: AnnData,
    adata_spatial: AnnData,
    *,
    method: str,
    noise_std: float = 10.0,
    fixed_augmentation: bool = True,
    leiden_resolution: float = 0.4,
    leiden_random_state: int = 0,
    preprocessing: str = GENE_LOCAL_PREPROCESSING,
) -> PreparedData:
    """Preprocess both domains; new runs use a gene-local, mask-safe scale."""

    method_key = method.strip().lower()
    if method_key not in {"stdiff", "imdt"}:
        raise ValueError("method must be 'stdiff' or 'imdt'.")
    if preprocessing not in {GENE_LOCAL_PREPROCESSING, LEGACY_PREPROCESSING}:
        raise ValueError(f"Unknown preprocessing mode: {preprocessing!r}.")

    validate_pair(
        adata_seq,
        adata_spatial,
        require_spatial_coords=(method_key == "imdt"),
    )
    seq = data_augment(
        adata_seq.copy(), fixed=fixed_augmentation, noise_std=noise_std
    )
    spatial = adata_spatial.copy()

    if preprocessing == LEGACY_PREPROCESSING:
        sc.pp.normalize_total(seq, target_sum=1e4)
        sc.pp.log1p(seq)
    else:
        seq = scale_gene_local(seq)
    seq_labels = None
    if method_key == "imdt":
        seq_labels = _cluster_leiden(
            seq,
            resolution=leiden_resolution,
            random_state=leiden_random_state,
        )
    if preprocessing == LEGACY_PREPROCESSING:
        seq = scale(seq)

    if preprocessing == LEGACY_PREPROCESSING:
        sc.pp.normalize_total(spatial, target_sum=1e4)
        sc.pp.log1p(spatial)
    else:
        spatial = scale_gene_local(spatial)
    spatial_labels = None
    if method_key == "imdt":
        spatial_labels = _cluster_leiden(
            spatial,
            resolution=leiden_resolution,
            random_state=leiden_random_state,
        )
    if preprocessing == LEGACY_PREPROCESSING:
        spatial = scale(spatial)

    return PreparedData(
        adata_seq=seq,
        adata_spatial=spatial,
        seq_array=_dense_float32(seq.X),
        spatial_array=_dense_float32(spatial.X),
        genes=np.asarray(adata_spatial.var_names, dtype=str),
        seq_labels=seq_labels,
        spatial_labels=spatial_labels,
        coordinates=get_spatial_coordinates(adata_spatial),
    )


def build_gene_folds(
    genes: Sequence[str],
    *,
    n_splits: int = 5,
    random_state: int = 0,
) -> list[GeneFold]:
    genes_array = np.asarray(genes, dtype=str)
    if len(genes_array) < n_splits:
        raise ValueError(
            f"At least {n_splits} genes are needed for {n_splits}-fold CV; "
            f"got {len(genes_array)}."
        )

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: list[GeneFold] = []
    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(genes_array), start=1
    ):
        folds.append(
            GeneFold(
                fold=fold_number,
                train_indices=train_indices.astype(np.int64),
                test_indices=test_indices.astype(np.int64),
                train_genes=tuple(genes_array[train_indices]),
                test_genes=tuple(genes_array[test_indices]),
            )
        )
    return folds


def build_fold_data(prepared: PreparedData, fold: GeneFold, *, method: str) -> FoldData:
    """Construct the fold mask and conditions exactly as in the prior scripts."""

    method_key = method.strip().lower()
    gene_count = prepared.spatial_array.shape[1]
    mask = np.ones((gene_count,), dtype=np.float32)
    mask[fold.test_indices] = 0.0

    seq_masked = prepared.seq_array * mask
    spatial_masked = prepared.spatial_array * mask
    seq_diffusion = prepared.seq_array * 2.0 - 1.0
    spatial_diffusion = prepared.spatial_array * 2.0 - 1.0
    seq_masked = seq_masked * 2.0 - 1.0
    spatial_masked = spatial_masked * 2.0 - 1.0

    if method_key == "stdiff":
        train_condition = seq_masked
        spatial_condition = spatial_masked
    elif method_key == "imdt":
        if prepared.seq_labels is None or prepared.spatial_labels is None:
            raise ValueError("IMD-T requires Leiden labels from preprocess_pair().")
        train_condition = prepared.seq_labels
        spatial_condition = prepared.spatial_labels
    else:
        raise ValueError("method must be 'stdiff' or 'imdt'.")

    return FoldData(
        mask=mask,
        train_expression=seq_diffusion,
        train_condition=train_condition,
        spatial_expression=spatial_diffusion,
        spatial_condition=spatial_condition,
        spatial_ground_truth=spatial_masked,
        test_indices=fold.test_indices,
    )


def get_data_loader(
    data_ary: np.ndarray,
    condition: np.ndarray,
    *,
    batch_size: int = 512,
    is_shuffle: bool = True,
    seed: int = 1202,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Build the common two-field DataLoader with a seeded CPU generator."""

    data_tensor = torch.from_numpy(np.asarray(data_ary, dtype=np.float32))
    condition_array = np.asarray(condition)
    if np.issubdtype(condition_array.dtype, np.integer):
        condition_tensor = torch.from_numpy(condition_array.astype(np.int64))
    else:
        condition_tensor = torch.from_numpy(condition_array.astype(np.float32))

    if len(data_tensor) != len(condition_tensor):
        raise ValueError(
            "Expression and condition arrays must have the same first dimension."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    dataset = TensorDataset(data_tensor, condition_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_shuffle,
        drop_last=False,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
    )


def fold_records(folds: Iterable[GeneFold]) -> list[dict[str, object]]:
    return [fold.as_dict() for fold in folds]


__all__ = [
    "DATASET_FILES",
    "DEFAULT_DATA_ROOT",
    "GENE_LOCAL_PREPROCESSING",
    "LEGACY_PREPROCESSING",
    "DatasetPaths",
    "FoldData",
    "GeneFold",
    "PreparedData",
    "build_fold_data",
    "build_gene_folds",
    "data_augment",
    "fold_records",
    "get_data_loader",
    "get_spatial_coordinates",
    "load_paired_h5ad",
    "preprocess_pair",
    "read_h5ad",
    "reindex",
    "resolve_dataset_paths",
    "scale",
    "scale_gene_local",
    "validate_pair",
]
