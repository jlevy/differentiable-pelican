from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def pick_device() -> torch.device:
    """
    Auto-detect the best available device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dir(path: Path) -> Path:
    """
    Ensure output directory exists.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
