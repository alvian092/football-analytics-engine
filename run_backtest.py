"""Command-line entry point for a walk-forward backtest.

Example:
    python run_backtest.py --input data/raw/matches.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.backtest import walk_forward_backtest
from engine.data import load_matches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a chronological pre-match 1X2 walk-forward backtest."
    )
    parser.add_argument("--input", required=True, help="Path to historical matches CSV.")
    parser.add_argument("--competition", default=None, help="Optional exact competition name.")
    parser.add_argument("--output-dir", default="data/processed/backtest",
                        help="Directory for metrics.json and predictions.csv.")
    parser.add_argument("--min-league-matches", type=int, default=30)
    parser.add_argument("--min-team-matches", type=int, default=3)
    parser.add_argument("--half-life-days", type=float, default=365.0)
    parser.add_argument("--prior-matches", type=float, default=5.0)
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument("--rho", type=float, default=None,
                        help="Optional Dixon-Coles low-score correction parameter.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_league_matches < 1:
        raise SystemExit("--min-league-matches must be at least 1")
    if args.min_team_matches < 1:
        raise SystemExit("--min-team-matches must be at least 1")
    if args.half_life_days <= 0:
        raise SystemExit("--half-life-days must be greater than 0")
    if args.prior_matches < 0:
        raise SystemExit("--prior-matches cannot be negative")
    if args.max_goals < 1:
        raise SystemExit("--max-goals must be at least 1")

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input CSV not found: {input_path}")

    matches = load_matches(input_path)
    result = walk_forward_backtest(
        matches,
        competition=args.competition,
        half_life_days=args.half_life_days,
        prior_matches=args.prior_matches,
        min_league_matches=args.min_league_matches,
        min_team_matches=args.min_team_matches,
        max_goals=args.max_goals,
        rho=args.rho,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"

    # The CSV contains one row per evaluated fixture. For insufficient history,
    # write a header-only file rather than fabricating forecast rows.
    import pandas as pd

    pd.DataFrame(result["predictions"]).to_csv(predictions_path, index=False)
    summary = {key: value for key, value in result.items() if key != "predictions"}
    summary["input_file"] = str(input_path)
    summary["competition_filter"] = args.competition
    summary["settings"] = {
        "min_league_matches": args.min_league_matches,
        "min_team_matches": args.min_team_matches,
        "half_life_days": args.half_life_days,
        "prior_matches": args.prior_matches,
        "max_goals": args.max_goals,
        "rho": args.rho,
    }
    metrics_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Status: {result['status']}")
    print(f"Evaluated fixtures: {result['evaluated_matches']}")
    print(f"Skipped fixtures (insufficient history): {result['skipped_matches']}")
    if result["status"] == "ok":
        print("Model:", json.dumps(result["model"], ensure_ascii=False))
        print("Baseline:", json.dumps(result["baseline"], ensure_ascii=False))
    print(f"Predictions CSV: {predictions_path}")
    print(f"Metrics JSON: {metrics_path}")
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
