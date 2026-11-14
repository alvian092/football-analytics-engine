"""End-to-end pre-match prediction pipeline."""
from __future__ import annotations

import pandas as pd

from engine.poisson import match_probabilities, score_matrix
from engine.strength import estimate_lambdas


def predict_fixture(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    as_of,
    *,
    competition: str | None = None,
    half_life_days: float = 365.0,
    prior_matches: float = 5.0,
    min_league_matches: int = 30,
    min_team_matches: int = 3,
    max_goals: int = 10,
    rho: float | None = None,
) -> dict:
    """Estimate lambdas and return pre-match scoreline/1X2 probabilities.

    Only historical matches strictly earlier than the cutoff are eligible.
    If the estimator has insufficient history, probability fields are None
    and the caller should treat the fixture as unavailable / NO BET.
    """
    result = estimate_lambdas(
        matches,
        home_team,
        away_team,
        as_of,
        competition=competition,
        half_life_days=half_life_days,
        prior_matches=prior_matches,
        min_league_matches=min_league_matches,
        min_team_matches=min_team_matches,
    )
    result["home_team"] = home_team.strip() if isinstance(home_team, str) else home_team
    result["away_team"] = away_team.strip() if isinstance(away_team, str) else away_team
    result["max_goals"] = max_goals
    result["rho"] = rho

    if result["status"] != "ok":
        result["probabilities"] = None
        result["score_matrix"] = None
        return result

    matrix = score_matrix(
        result["lambda_home"],
        result["lambda_away"],
        max_goals=max_goals,
        rho=rho,
    )
    result["probabilities"] = match_probabilities(matrix)
    result["score_matrix"] = matrix.tolist()
    return result
