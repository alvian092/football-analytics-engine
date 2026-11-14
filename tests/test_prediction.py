import pandas as pd

from engine.predict import predict_fixture


def sample_history():
    rows = []
    for i in range(38):
        rows.append({
            "date": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=i),
            "competition": "Test League",
            "season": "2025",
            "home_team": "Home FC" if i % 2 == 0 else f"Host {i}",
            "away_team": f"Visitor {i}" if i % 2 == 0 else "Away FC",
            "home_goals": 2 if i % 2 == 0 else 1,
            "away_goals": 1 if i % 2 == 0 else 0,
        })
    return pd.DataFrame(rows)


def test_prediction_pipeline_returns_probabilities():
    result = predict_fixture(
        sample_history(), "Home FC", "Away FC", "2025-03-01",
        competition="Test League", min_league_matches=30, min_team_matches=3,
        rho=-0.05,
    )
    assert result["status"] == "ok"
    assert result["lambda_home"] > 0
    assert result["lambda_away"] > 0
    probs = result["probabilities"]
    assert abs(probs["home_win"] + probs["draw"] + probs["away_win"] - 1) < 1e-9
    assert len(result["score_matrix"]) == 11


def test_prediction_pipeline_does_not_predict_without_enough_history():
    result = predict_fixture(
        sample_history(), "Home FC", "Away FC", "2025-01-05",
        competition="Test League", min_league_matches=30, min_team_matches=3,
    )
    assert result["status"] == "insufficient_data"
    assert result["probabilities"] is None
    assert result["score_matrix"] is None
