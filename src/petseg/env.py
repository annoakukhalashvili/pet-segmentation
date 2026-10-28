"""Helpers for inspecting the runtime environment and choosing a device."""

import platform
import sys

import torch


def select_device(preferred: str = "auto") -> torch.device:
    """Return the best available device, or the one explicitly requested.

    Args:
        preferred: "auto", "cuda", "mps" or "cpu". "auto" picks
            CUDA (NVIDIA GPU) first, then MPS (Apple Silicon GPU), then CPU.

    Raises:
        ValueError: if the name is unknown.
        RuntimeError: if the requested device is not available.
    """
    preferred = preferred.lower()
    if preferred not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unknown device '{preferred}'. Use auto, cuda, mps or cpu.")

    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preferred == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine.")
    if preferred == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available on this machine.")
    return torch.device(preferred)


def collect_env_info() -> dict[str, str]:
    """Collect version and hardware information useful for reproducibility."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "mps_available": str(torch.backends.mps.is_available()),
        "selected_device": str(select_device()),
    }