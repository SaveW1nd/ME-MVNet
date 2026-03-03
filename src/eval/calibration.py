"""Calibration helpers."""

from __future__ import annotations

import numpy as np


def expected_nf_from_probs(nf_probs: np.ndarray, nf_values: list[int]) -> np.ndarray:
    """Compute E[NF] from class probabilities."""
    vals = np.asarray(nf_values, dtype=np.float32).reshape(1, -1)
    return np.sum(nf_probs * vals, axis=1)
