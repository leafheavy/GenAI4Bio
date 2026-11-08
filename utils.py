"""Small utilities shared by the research pipeline.

Keep this module independent from the other project modules.  Only functions
that are reused by more than one stage belong here.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> torch.Generator:
    """Seed Python, NumPy, PyTorch, CUDA, and a CPU DataLoader generator."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an explicit device, defaulting to CUDA when available."""

    resolved = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({resolved}), but CUDA is unavailable.")
    return resolved


def ensure_output_dir(
    path: str | os.PathLike[str], *, allow_existing: bool = False
) -> Path:
    """Create an output directory without silently overwriting an old run."""

    output_dir = Path(path).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new directory or pass allow_existing=True explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(data: Any, path: str | os.PathLike[str]) -> Path:
    """Write human-readable UTF-8 JSON, creating the parent directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    return destination


def load_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def environment_snapshot() -> dict[str, Any]:
    """Collect the compact software metadata needed to interpret a run."""

    packages: dict[str, str] = {}
    for name in (
        "anndata",
        "einops",
        "leidenalg",
        "matplotlib",
        "numpy",
        "pandas",
        "scanpy",
        "scikit-learn",
        "scipy",
        "seaborn",
        "timm",
        "torch",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "packages": packages,
    }


__all__ = [
    "ensure_output_dir",
    "environment_snapshot",
    "load_json",
    "resolve_device",
    "save_json",
    "set_seed",
    "sha256_file",
]
