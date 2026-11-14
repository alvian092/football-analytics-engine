"""Chronological walk-forward evaluation for pre-match 1X2 forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.data import validate_matches
from engine.predict import predict_fixture

OUTCOMES = ("home_win", "draw", "away_win")


def _actual_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def _multiclass_brier(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean per-match sum of squared errors across the three classes."""
    return float(np.mean(np.sum((probabilities - outcomes) ** 2, axis=1)))


def _multiclass_log_loss(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean negative log probability assigned to the realized class."""
    chosen = np.sum(probabilities * outcomes, axis=1)
    return float(-np.mean(np.log(np.clip(chosen, 1e-15, 1.0))))


def _smoothed_baseline(training: pd.DataFrame, alpha: float = 1.0) -> dict[str, float]:
    """Laplace-smoothed historical 1X2 frequencies from prior fixtures only."""
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    home = int((training["home_goals"] > training["away_goals"]).sum())
    draw = int((training["home_goals"] == training["away_goals"]).sum())
    away = int((training["home_goals"] < training["away_goals"]).sum())
    denominator = len(training) + 3.0 * alpha
    return {
        "home_win": (home + alpha) / denominator,
        "draw": (draw + alpha) / denominator,
        "away_win": (away + alpha) / denominator,
    }


def walk_forward_backtest(
    matches: pd.DataFrame,
    *,
    competition: str | None = None,
    half_life_days: float = 365.0,
    prior_matches: float = 5.0,
    min_league_matches: int = 30,
    min_team_matches: int = 3,
    max_goals: int = 10,
    rho: float | None = None,
    baseline_alpha: float = 1.0,
) -> dict:
    """Evaluate each fixture using only matches strictly earlier than its date.

    Fixtures on the same date are evaluated against the same prior history;
    none of that date's results can influence another fixture on that date.
    """
    frame = validate_matches(matches)
    if competition is not None:
        if not isinstance(competition, str) or not competition.strip():
            raise ValueError("competition must be a non-empty string or None")
        frame = frame.loc[frame["competition"] == competition.strip()].copy()
    if frame.empty:
        raise ValueError("no matches available for the selected competition")
    if baseline_alpha <= 0:
        raise ValueError("baseline_alpha must be > 0")

    evaluated: list[dict] = []
    skipped = 0
    for fixture in frame.itertuples(index=False):
        cutoff = fixture.date
        training = frame.loc[frame["date"] < cutoff].copy()
        league_training = training.loc[training["competition"] == fixture.competition].copy()
        prediction = predict_fixture(
            league_training,
            fixture.home_team,
            fixture.away_team,
            cutoff,
            competition=competition or fixture.competition,
            half_life_days=half_life_days,
            prior_matches=prior_matches,
            min_league_matches=min_league_matches,
            min_team_matches=min_team_matches,
            max_goals=max_goals,
            rho=rho,
        )
        if prediction["status"] != "ok" or prediction["probabilities"] is None:
            skipped += 1
            continue

        probs = prediction["probabilities"]
        model_probs = np.array([probs[key] for key in OUTCOMES], dtype=float)
        model_probs = model_probs / model_probs.sum()
        baseline = _smoothed_baseline(league_training, baseline_alpha)
        baseline_probs = np.array([baseline[key] for key in OUTCOMES], dtype=float)
        actual = _actual_outcome(int(fixture.home_goals), int(fixture.away_goals))
        one_hot = np.array([1.0 if key == actual else 0.0 for key in OUTCOMES])
        evaluated.append({
            "date": cutoff.isoformat(),
            "competition": fixture.competition,
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "home_goals": int(fixture.home_goals),
            "away_goals": int(fixture.away_goals),
            "actual": actual,
            "historical_matches": prediction["historical_matches"],
            "home_win_probability": float(model_probs[0]),
            "draw_probability": float(model_probs[1]),
            "away_win_probability": float(model_probs[2]),
            "baseline_home_win_probability": float(baseline_probs[0]),
            "baseline_draw_probability": float(baseline_probs[1]),
            "baseline_away_win_probability": float(baseline_probs[2]),
            "_model": model_probs,
            "_baseline": baseline_probs,
            "_actual_vector": one_hot,
        })

    if not evaluated:
        return {
            "status": "insufficient_data",
            "evaluated_matches": 0,
            "skipped_matches": skipped,
            "model": None,
            "baseline": None,
            "predictions": [],
            "reason": "no_fixtures_had_sufficient_prior_history",
        }

    model = np.vstack([row["_model"] for row in evaluated])
    baseline = np.vstack([row["_baseline"] for row in evaluated])
    actuals = np.vstack([row["_actual_vector"] for row in evaluated])

    def metrics(probabilities: np.ndarray) -> dict[str, float]:
        return {
            "brier_score": _multiclass_brier(probabilities, actuals),
            "log_loss": _multiclass_log_loss(probabilities, actuals),
        }

    public_rows = [
        {key: value for key, value in row.items()
         if not key.startswith("_")}
        for row in evaluated
    ]
    return {
        "status": "ok",
        "evaluated_matches": len(evaluated),
        "skipped_matches": skipped,
        "model": metrics(model),
        "baseline": metrics(baseline),
        "predictions": public_rows,
        "interpretation": (
            "Lower Brier Score and Log Loss are better. Compare model with "
            "baseline; this backtest does not establish profitability."
        ),
    }
