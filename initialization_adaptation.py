"""Trainable test-time initial distributions for differentiable flow sampling.

The first implementation deliberately changes only the Gaussian mean.  The
source flow is expected to stay frozen and to receive samples returned by this
module, so gradients can travel from a downstream loss back to ``mu``.
"""

from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class InitializationAdapter(nn.Module, ABC):
    """Common interface for future test-time initialization families."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of one initial state sample."""

    @abstractmethod
    def sample_noise(
        self,
        batch_size: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Draw parameter-independent base noise."""

    @abstractmethod
    def sample(
        self,
        batch_size: int,
        *,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return a reparameterized initial state batch."""

    @abstractmethod
    def regularization(self) -> torch.Tensor:
        """Return divergence from the frozen source initialization."""

    def adaptation_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return only parameters intended for test-time optimization."""

        parameters = tuple(parameter for parameter in self.parameters() if parameter.requires_grad)
        if not parameters:
            raise RuntimeError("The initialization adapter has no trainable parameters.")
        return parameters


class MeanOnlyGaussianInitialization(InitializationAdapter):
    """Dataset-level ``N(mu, I)`` initialization with one shared global mean.

    ``mu`` has shape ``(1, dimension)`` and broadcasts over target spots.  It is
    initialized to zero, so sampling starts exactly from the source prior
    ``N(0, I)``.
    """

    def __init__(
        self,
        dimension: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("dimension must be an integer.")
        if dimension < 1:
            raise ValueError("dimension must be positive.")

        self._dimension = dimension
        factory_kwargs = {"device": device, "dtype": dtype}
        self.mu = nn.Parameter(torch.zeros((1, dimension), **factory_kwargs))

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def mean(self) -> torch.Tensor:
        """Expose the current global mean without making a detached copy."""

        return self.mu

    def sample_noise(
        self,
        batch_size: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        noise_device = self.mu.device
        if generator is not None:
            generator_device = torch.device(generator.device)
            if generator_device.type != self.mu.device.type:
                noise_device = generator_device
        noise = torch.randn(
            (batch_size, self.dimension),
            device=noise_device,
            dtype=self.mu.dtype,
            generator=generator,
        )
        return noise.to(self.mu.device)

    def sample(
        self,
        batch_size: int,
        *,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = self.sample_noise(batch_size, generator=generator)
        else:
            if not isinstance(noise, torch.Tensor):
                raise TypeError("noise must be a torch.Tensor.")
            expected_shape = (batch_size, self.dimension)
            if tuple(noise.shape) != expected_shape:
                raise ValueError(
                    f"noise must have shape {expected_shape}, got {tuple(noise.shape)}."
                )
            noise = noise.to(device=self.mu.device, dtype=self.mu.dtype)
        return self.mu + noise

    def regularization(self) -> torch.Tensor:
        """KL(N(mu, I) || N(0, I)) = 0.5 * ||mu||^2."""

        return 0.5 * self.mu.square().sum()

    def extra_repr(self) -> str:
        return f"dimension={self.dimension}, device={self.mu.device}, dtype={self.mu.dtype}"


def _resolve_cli_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({device}), but CUDA is unavailable.")
    return device


def _load_cli_tensor(
    path: str,
    *,
    key: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Tensor file not found: {source}")
    if source.suffix.lower() == ".npy":
        value = np.load(source, allow_pickle=False)
    elif source.suffix.lower() in {".pt", ".pth"}:
        try:
            value = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:
            value = torch.load(source, map_location="cpu")
        if isinstance(value, dict):
            if key not in value:
                raise KeyError(f"{source} does not contain the key {key!r}.")
            value = value[key]
    else:
        raise ValueError("Tensor files must use .pt, .pth, or .npy.")
    return torch.as_tensor(value, device=device, dtype=dtype)


def _resolve_dataset_dimension(
    dataset: str,
    data_root: Optional[str],
    sc_path: Optional[str],
    spatial_path: Optional[str],
) -> tuple[int, str, str, str]:
    """Resolve ``G`` through the existing paired-H5AD data contract."""

    if (sc_path is None) != (spatial_path is None):
        raise ValueError("--sc-path and --spatial-path must be provided together.")
    try:
        from .Dataloader import (
            DEFAULT_DATA_ROOT,
            load_paired_h5ad,
            resolve_dataset_paths,
        )
    except ImportError:
        from Dataloader import (
            DEFAULT_DATA_ROOT,
            load_paired_h5ad,
            resolve_dataset_paths,
        )

    if sc_path is None:
        resolved = resolve_dataset_paths(dataset, data_root or DEFAULT_DATA_ROOT)
        sc_file = resolved.sc_h5ad
        spatial_file = resolved.spatial_h5ad
        dataset_name = resolved.dataset
    else:
        sc_file = Path(sc_path).expanduser()
        spatial_file = Path(spatial_path).expanduser()
        dataset_name = dataset
    sc_file = Path(sc_file).expanduser().resolve()
    spatial_file = Path(spatial_file).expanduser().resolve()
    adata_seq, _ = load_paired_h5ad(sc_file, spatial_file)
    return int(adata_seq.n_vars), str(sc_file), str(spatial_file), dataset_name


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample the mean-only Gaussian initialization N(mu, I)."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--sc-path", default=None)
    parser.add_argument("--spatial-path", default=None)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-noise-samples", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=1202)
    parser.add_argument(
        "--mu",
        default=None,
        help="Optional .pt/.npy tensor with shape [d] or [1,d]; default is zero.",
    )
    parser.add_argument(
        "--noise-samples",
        default=None,
        help="Optional .pt/.npy tensor with shape [B,d] or [K,B,d].",
    )
    parser.add_argument("--output", required=True, help="Output .pt file.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.num_noise_samples < 1:
        raise ValueError("num_noise_samples must be positive.")
    output = Path(args.output)
    if output.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("--output must end in .pt or .pth.")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_cli_device(args.device)
    dtype = getattr(torch, args.dtype)
    dimension, sc_path, spatial_path, dataset = _resolve_dataset_dimension(
        args.dataset,
        args.data_root,
        args.sc_path,
        args.spatial_path,
    )
    adapter = MeanOnlyGaussianInitialization(
        dimension, device=device, dtype=dtype
    )
    if args.mu is not None:
        mu = _load_cli_tensor(
            args.mu, key="mu", device=device, dtype=dtype
        )
        if mu.ndim == 1:
            mu = mu.unsqueeze(0)
        if tuple(mu.shape) != tuple(adapter.mu.shape):
            raise ValueError(
                f"mu must have shape {tuple(adapter.mu.shape)}, got {tuple(mu.shape)}."
            )
        with torch.no_grad():
            adapter.mu.copy_(mu)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    if args.noise_samples is None:
        noise_samples = torch.stack(
            [
                adapter.sample_noise(args.batch_size, generator=generator)
                for _ in range(args.num_noise_samples)
            ]
        )
    else:
        noise_samples = _load_cli_tensor(
            args.noise_samples,
            key="noise_samples",
            device=device,
            dtype=dtype,
        )
        if noise_samples.ndim == 2:
            noise_samples = noise_samples.unsqueeze(0)
        expected = (args.num_noise_samples, args.batch_size, dimension)
        if tuple(noise_samples.shape) != expected:
            raise ValueError(
                f"noise_samples must have shape {expected}, "
                f"got {tuple(noise_samples.shape)}."
            )

    initial_states = noise_samples + adapter.mu.unsqueeze(0)
    payload = {
        "mu": adapter.mu.detach().cpu(),
        "noise_samples": noise_samples.detach().cpu(),
        "initial_states": initial_states.detach().cpu(),
        "kl_regularization": adapter.regularization().detach().cpu(),
        "dataset": dataset,
        "dimension": dimension,
        "seed": args.seed,
    }
    torch.save(payload, output)
    metadata = {
        "output": str(output),
        "dataset": dataset,
        "dimension": dimension,
        "sc_h5ad": sc_path,
        "spatial_h5ad": spatial_path,
        "batch_size": args.batch_size,
        "num_noise_samples": args.num_noise_samples,
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "kl_regularization": float(adapter.regularization().detach().item()),
    }
    metadata_path = output.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


__all__ = [
    "InitializationAdapter",
    "MeanOnlyGaussianInitialization",
]


if __name__ == "__main__":
    main()
