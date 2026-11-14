import pandas as pd

from engine.backtest import walk_forward_backtest


def synthetic_matches(n=48):
    teams = [f"Team {i}" for i in range(6)]
    rows = []
    start = pd.Timestamp("2025-01-01", tz="UTC")
    for i in range(n):
        home_idx = i % len(teams)
        away_idx = (home_idx + 1 + (i // len(teams)) % 4) % len(teams)
        rows.append({
            "date": start + pd.Timedelta(days=i),
            "competition": "Synthetic League",
            "season": "2025",
            "home_team": teams[home_idx],
            "away_team": teams[away_idx],
            "home_goals": (i * 3) % 4,
            "away_goals": (i * 5 + 1) % 3,
        })
    return pd.DataFrame(rows)


def test_walk_forward_backtest_returns_metrics_and_predictions():
    result = walk_forward_backtest(
        synthetic_matches(),
        competition="Synthetic League",
        min_league_matches=10,
        min_team_matches=2,
    )
    assert result["status"] == "ok"
    assert result["evaluated_matches"] > 0
    assert result["skipped_matches"] > 0
    assert set(result["model"]) == {"brier_score", "log_loss"}
    assert set(result["baseline"]) == {"brier_score", "log_loss"}
    assert 0 <= result["model"]["brier_score"] <= 2
    assert result["model"]["log_loss"] > 0
    assert result["predictions"][0]["historical_matches"] < 48


def test_backtest_does_not_use_same_day_or_future_matches():
    frame = synthetic_matches()
    frame.loc[1, "date"] = frame.loc[0, "date"]
    result = walk_forward_backtest(
        frame,
        competition="Synthetic League",
        min_league_matches=5,
        min_team_matches=1,
    )
    predictions = result["predictions"]
    same_day = [
        row for row in predictions
        if row["date"] == frame.loc[0, "date"].isoformat()
    ]
    assert same_day == []
    for row in predictions:
        fixture_date = pd.to_datetime(row["date"], utc=True)
        expected_prior = int((frame["date"] < fixture_date).sum())
        assert row["historical_matches"] == expected_prior


def test_backtest_reports_insufficient_history_without_fake_metrics():
    result = walk_forward_backtest(
        synthetic_matches(5),
        competition="Synthetic League",
        min_league_matches=10,
        min_team_matches=2,
    )
    assert result["status"] == "insufficient_data"
    assert result["evaluated_matches"] == 0
    assert result["model"] is None
    assert result["baseline"] is None
