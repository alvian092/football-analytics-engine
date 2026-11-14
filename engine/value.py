"""Fair-odds, expected-value, and Kelly utilities."""
from __future__ import annotations

import math


def fair_odds(probability: float) -> float:
    if not 0 < probability <= 1 or not math.isfinite(probability):
        raise ValueError("probability must be in (0, 1]")
    return 1.0 / probability


def expected_value(probability: float, decimal_odds: float) -> float:
    """Expected net return per 1 unit stake for a no-push binary outcome."""
    if not 0 <= probability <= 1 or not math.isfinite(probability):
        raise ValueError("probability must be between 0 and 1")
    if decimal_odds <= 1 or not math.isfinite(decimal_odds):
        raise ValueError("decimal_odds must be finite and > 1")
    return probability * decimal_odds - 1.0


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    """Full Kelly fraction; returns 0 when there is no positive edge."""
    if not 0 <= probability <= 1 or not math.isfinite(probability):
        raise ValueError("probability must be between 0 and 1")
    if decimal_odds <= 1 or not math.isfinite(decimal_odds):
        raise ValueError("decimal_odds must be finite and > 1")
    b = decimal_odds - 1.0
    q = 1.0 - probability
    return max(0.0, (b * probability - q) / b)


def fractional_kelly(probability: float, decimal_odds: float, fraction: float = 0.25) -> float:
    if not 0 < fraction <= 1 or not math.isfinite(fraction):
        raise ValueError("fraction must be in (0, 1]")
    return kelly_fraction(probability, decimal_odds) * fraction
