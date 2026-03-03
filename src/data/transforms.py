"""Lightweight tensor transforms."""

from __future__ import annotations

import numpy as np


def standardize_iq(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-sample channel-wise normalization.

    Args:
        x: float32 array with shape (2, N).
        eps: numerical stability.

    Returns:
        Normalized float32 array with shape (2, N).
    """
    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError(f"Expected shape (2, N), got {x.shape}")
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + eps
    return ((x - mean) / std).astype(np.float32)
