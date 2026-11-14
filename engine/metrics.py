"""Probability forecast evaluation metrics."""
from __future__ import annotations

import numpy as np


def brier_score(probabilities, outcomes) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probabilities and outcomes must have equal non-empty shapes")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("inputs must be finite")
    if np.any((p < 0) | (p > 1)) or np.any((y < 0) | (y > 1)):
        raise ValueError("inputs must be in [0, 1]")
    return float(np.mean((p - y) ** 2))


def log_loss(probabilities, outcomes, eps: float = 1e-15) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probabilities and outcomes must have equal non-empty shapes")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("inputs must be finite")
    if np.any((p < 0) | (p > 1)) or np.any((y < 0) | (y > 1)):
        raise ValueError("inputs must be in [0, 1]")
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
