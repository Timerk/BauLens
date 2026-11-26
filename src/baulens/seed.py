"""Seed the random number generators used by the pipelines."""

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and optional PyTorch, and require deterministic torch operations."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
