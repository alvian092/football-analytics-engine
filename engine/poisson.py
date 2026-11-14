"""Poisson scoreline probabilities and Dixon-Coles correction."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def poisson_pmf(k: int, rate: float) -> float:
    """Return P(X=k) for a Poisson variable with non-negative rate."""
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("rate must be finite and non-negative")
    if rate == 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-rate + k * math.log(rate) - math.lgamma(k + 1))


def score_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 10,
    rho: Optional[float] = None,
) -> np.ndarray:
    """Build a normalized score matrix; rows=home goals, columns=away goals.

    If rho is supplied, apply the Dixon-Coles correction to 0-0, 0-1,
    1-0 and 1-1 before renormalizing the truncated matrix.
    """
    if not math.isfinite(lambda_home) or lambda_home < 0:
        raise ValueError("lambda_home must be finite and non-negative")
    if not math.isfinite(lambda_away) or lambda_away < 0:
        raise ValueError("lambda_away must be finite and non-negative")
    if not isinstance(max_goals, int) or max_goals < 2:
        raise ValueError("max_goals must be an integer >= 2")
    if rho is not None and not math.isfinite(rho):
        raise ValueError("rho must be finite")

    home = np.array([poisson_pmf(k, lambda_home) for k in range(max_goals + 1)])
    away = np.array([poisson_pmf(k, lambda_away) for k in range(max_goals + 1)])
    matrix = np.outer(home, away)

    if rho is not None:
        corrections = {
            (0, 0): 1 - lambda_home * lambda_away * rho,
            (0, 1): 1 + lambda_home * rho,
            (1, 0): 1 + lambda_away * rho,
            (1, 1): 1 - rho,
        }
        for (h, a), factor in corrections.items():
            if factor < 0:
                raise ValueError(
                    "rho produces a negative Dixon-Coles probability; "
                    "use a less extreme rho"
                )
            matrix[h, a] *= factor

    total = float(matrix.sum())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("score matrix has invalid total probability")
    return matrix / total


def match_probabilities(matrix: np.ndarray) -> dict[str, float]:
    """Derive 1X2, BTTS, and expected goals from a score matrix."""
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("matrix must be a non-empty 2D array")
    if np.any(matrix < 0) or not np.isfinite(matrix).all():
        raise ValueError("matrix must contain finite, non-negative values")
    total = float(matrix.sum())
    if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("matrix must be normalized to 1")

    home_win = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())
    btts_yes = float(matrix[1:, 1:].sum())
    home_goals = float(sum(h * matrix[h, a] for h in range(matrix.shape[0]) for a in range(matrix.shape[1])))
    away_goals = float(sum(a * matrix[h, a] for h in range(matrix.shape[0]) for a in range(matrix.shape[1])))

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "btts_yes": btts_yes,
        "btts_no": 1.0 - btts_yes,
        "expected_home_goals_truncated": home_goals,
        "expected_away_goals_truncated": away_goals,
    }


def total_goals_probabilities(matrix: np.ndarray, line: float) -> dict[str, float]:
    """Return over/under/push probabilities for a total-goals line.

    Half-goal lines have no push. Integer lines return an explicit push
    probability. Quarter-lines are intentionally deferred to settlement-aware
    Asian-total calculations in a later milestone.
    """
    if not math.isfinite(line) or line < 0:
        raise ValueError("line must be finite and non-negative")
    if abs(line * 2 - round(line * 2)) > 1e-9 or abs(line - round(line)) > 1e-9 and abs(line * 2 - round(line * 2)) > 1e-9:
        raise ValueError("only integer and half-goal lines are supported in this version")

    over = under = push = 0.0
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            goals = h + a
            p = float(matrix[h, a])
            if goals > line:
                over += p
            elif goals < line:
                under += p
            else:
                push += p
    return {"over": over, "under": under, "push": push}
