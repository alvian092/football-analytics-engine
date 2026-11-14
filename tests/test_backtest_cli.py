import json

import pandas as pd

from run_backtest import main


def test_cli_writes_metrics_and_prediction_csv(tmp_path):
    rows = []
    teams = [f"Club {i}" for i in range(6)]
    start = pd.Timestamp("2025-01-01", tz="UTC")
    for i in range(48):
        home_idx = i % 6
        away_idx = (home_idx + 1 + (i // 6) % 4) % 6
        rows.append({
            "date": start + pd.Timedelta(days=i),
            "competition": "CLI League",
            "season": "2025",
            "home_team": teams[home_idx],
            "away_team": teams[away_idx],
            "home_goals": (i * 3) % 4,
            "away_goals": (i * 5 + 1) % 3,
        })
    input_path = tmp_path / "matches.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    output_dir = tmp_path / "report"

    exit_code = main([
        "--input", str(input_path),
        "--competition", "CLI League",
        "--output-dir", str(output_dir),
        "--min-league-matches", "10",
        "--min-team-matches", "2",
    ])

    assert exit_code == 0
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.csv"
    assert metrics_path.is_file()
    assert predictions_path.is_file()
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    assert summary["status"] == "ok"
    assert summary["evaluated_matches"] == len(predictions)
    assert "brier_score" in summary["model"]
    assert "log_loss" in summary["baseline"]
