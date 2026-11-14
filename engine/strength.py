"""Transparent historical team-strength and pre-match lambda estimation."""
from __future__ import annotations

import math

import pandas as pd

from engine.data import validate_matches


def _weighted_mean(values: pd.Series, weights: pd.Series, prior_mean: float, prior_matches: float) -> float:
    denominator = float(weights.sum()) + prior_matches
    if denominator <= 0:
        return float(prior_mean)
    return float(((values.astype(float) * weights).sum() + prior_mean * prior_matches) / denominator)


def estimate_lambdas(
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
) -> dict:
    """Estimate expected goals from matches strictly before the cutoff.

    Team venue-specific scoring/conceding rates are shrunk toward competition
    averages. Recency weights use an exponential half-life. If the league or
    either team's relevant venue sample is too small, no lambdas are returned.
    """
    if not isinstance(home_team, str) or not home_team.strip():
        raise ValueError("home_team must be a non-empty string")
    if not isinstance(away_team, str) or not away_team.strip():
        raise ValueError("away_team must be a non-empty string")
    home_team, away_team = home_team.strip(), away_team.strip()
    if home_team == away_team:
        raise ValueError("home_team and away_team must be different")
    if not math.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("half_life_days must be finite and > 0")
    if not math.isfinite(prior_matches) or prior_matches < 0:
        raise ValueError("prior_matches must be finite and >= 0")
    if not isinstance(min_league_matches, int) or min_league_matches < 1:
        raise ValueError("min_league_matches must be an integer >= 1")
    if not isinstance(min_team_matches, int) or min_team_matches < 1:
        raise ValueError("min_team_matches must be an integer >= 1")

    frame = validate_matches(matches)
    cutoff = pd.to_datetime(as_of, utc=True, errors="coerce")
    if pd.isna(cutoff):
        raise ValueError("as_of must be a valid date or timestamp")

    frame = frame.loc[frame["date"] < cutoff].copy()
    if competition is not None:
        if not isinstance(competition, str) or not competition.strip():
            raise ValueError("competition must be a non-empty string or None")
        frame = frame.loc[frame["competition"] == competition.strip()].copy()

    result = {
        "status": "insufficient_data",
        "lambda_home": None,
        "lambda_away": None,
        "historical_matches": int(len(frame)),
        "home_team_home_matches": 0,
        "away_team_away_matches": 0,
        "competition": competition,
        "as_of": cutoff.isoformat(),
    }
    if len(frame) < min_league_matches:
        result["reason"] = "not_enough_prior_league_matches"
        return result

    age_days = (cutoff - frame["date"]).dt.total_seconds() / 86400.0
    weights = pd.Series(
        2.0 ** (-age_days.to_numpy() / half_life_days),
        index=frame.index,
        dtype=float,
    )
    league_home = _weighted_mean(frame["home_goals"], weights, 0.0, 0.0)
    league_away = _weighted_mean(frame["away_goals"], weights, 0.0, 0.0)
    if league_home <= 0 or league_away <= 0:
        result["reason"] = "invalid_league_goal_baseline"
        return result

    home_rows = frame.loc[frame["home_team"] == home_team]
    away_rows = frame.loc[frame["away_team"] == away_team]
    home_count, away_count = len(home_rows), len(away_rows)
    result.update({
        "home_team_home_matches": int(home_count),
        "away_team_away_matches": int(away_count),
        "league_home_goals_avg": league_home,
        "league_away_goals_avg": league_away,
    })
    if home_count < min_team_matches or away_count < min_team_matches:
        result["reason"] = "not_enough_team_home_away_history"
        return result

    home_weights = weights.loc[home_rows.index]
    away_weights = weights.loc[away_rows.index]

    home_scored = _weighted_mean(home_rows["home_goals"], home_weights, league_home, prior_matches)
    home_conceded = _weighted_mean(home_rows["away_goals"], home_weights, league_away, prior_matches)
    away_scored = _weighted_mean(away_rows["away_goals"], away_weights, league_away, prior_matches)
    away_conceded = _weighted_mean(away_rows["home_goals"], away_weights, league_home, prior_matches)

    home_attack = home_scored / league_home
    home_defence = home_conceded / league_away
    away_attack = away_scored / league_away
    away_defence = away_conceded / league_home

    lambda_home = league_home * home_attack * away_defence
    lambda_away = league_away * away_attack * home_defence
    if not all(math.isfinite(x) and x > 0 for x in (lambda_home, lambda_away)):
        result["reason"] = "invalid_lambda_estimate"
        return result

    result.update({
        "status": "ok",
        "reason": None,
        "lambda_home": float(lambda_home),
        "lambda_away": float(lambda_away),
        "home_attack_strength": float(home_attack),
        "home_defence_strength": float(home_defence),
        "away_attack_strength": float(away_attack),
        "away_defence_strength": float(away_defence),
    })
    return result
