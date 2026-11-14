import pandas as pd
import pytest

from engine.strength import estimate_lambdas


def history():
    rows = []
    for i in range(30):
        rows.append({
            "date": f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            "competition": "Test League",
            "season": "2025",
            "home_team": "Home FC" if i % 2 == 0 else "Other Home",
            "away_team": "Away FC" if i % 2 == 0 else "Other Away",
            "home_goals": 2 if i % 2 == 0 else 1,
            "away_goals": 1 if i % 2 == 0 else 0,
        })
    for i in range(4):
        rows.append({
            "date": f"2025-03-{i + 1:02d}",
            "competition": "Test League",
            "season": "2025",
            "home_team": "Home FC",
            "away_team": f"Visitor {i}",
            "home_goals": 2,
            "away_goals": 1,
        })
        rows.append({
            "date": f"2025-03-{i + 5:02d}",
            "competition": "Test League",
            "season": "2025",
            "home_team": f"Host {i}",
            "away_team": "Away FC",
            "home_goals": 1,
            "away_goals": 1,
        })
    return pd.DataFrame(rows)


def test_estimator_uses_prior_matches_and_returns_positive_lambdas():
    result = estimate_lambdas(
        history(), "Home FC", "Away FC", "2025-04-01",
        competition="Test League", min_league_matches=30, min_team_matches=3,
    )
    assert result["status"] == "ok"
    assert result["lambda_home"] > 0
    assert result["lambda_away"] > 0
    assert result["historical_matches"] == 38


def test_estimator_excludes_same_day_and_future_matches():
    result = estimate_lambdas(
        history(), "Home FC", "Away FC", "2025-03-01",
        competition="Test League", min_league_matches=1, min_team_matches=1,
    )
    assert result["historical_matches"] < 38
    assert result["status"] == "insufficient_data"


def test_estimator_requires_sufficient_league_history():
    result = estimate_lambdas(
        history(), "Home FC", "Away FC", "2025-04-01",
        competition="Test League", min_league_matches=100,
    )
    assert result["status"] == "insufficient_data"
    assert result["lambda_home"] is None


def test_estimator_rejects_same_team():
    with pytest.raises(ValueError, match="different"):
        estimate_lambdas(history(), "Home FC", "Home FC", "2025-04-01")
