import math

import numpy as np
import pytest

from engine.metrics import brier_score, log_loss
from engine.poisson import match_probabilities, poisson_pmf, score_matrix, total_goals_probabilities
from engine.value import expected_value, fair_odds, fractional_kelly, kelly_fraction


def test_poisson_pmf_basic_values():
    assert poisson_pmf(0, 0) == 1
    assert poisson_pmf(1, 0) == 0
    assert poisson_pmf(0, 1) == pytest.approx(math.exp(-1))


def test_score_matrix_is_normalized():
    matrix = score_matrix(1.5, 1.1, max_goals=12)
    assert matrix.shape == (13, 13)
    assert matrix.sum() == pytest.approx(1.0)


def test_dixon_coles_matrix_is_normalized():
    matrix = score_matrix(1.5, 1.1, max_goals=12, rho=-0.08)
    assert matrix.sum() == pytest.approx(1.0)
    assert np.all(matrix >= 0)


def test_1x2_probabilities_sum_to_one():
    probs = match_probabilities(score_matrix(1.6, 0.9))
    assert probs["home_win"] + probs["draw"] + probs["away_win"] == pytest.approx(1.0)


def test_total_probabilities_include_push():
    result = total_goals_probabilities(score_matrix(1.4, 1.1), 2.0)
    assert result["over"] + result["under"] + result["push"] == pytest.approx(1.0)
    assert result["push"] > 0


def test_value_calculations():
    assert fair_odds(0.5) == pytest.approx(2.0)
    assert expected_value(0.5, 2.2) == pytest.approx(0.1)
    assert kelly_fraction(0.5, 2.0) == pytest.approx(0.0)
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
    assert fractional_kelly(0.6, 2.0, 0.25) == pytest.approx(0.05)


def test_metrics():
    assert brier_score([0.8, 0.2], [1, 0]) == pytest.approx(0.04)
    assert log_loss([0.8, 0.2], [1, 0]) > 0
