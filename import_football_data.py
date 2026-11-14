"""Convert manually downloaded Football-Data.co.uk CSV files for the engine."""
from __future__ import annotations

import argparse
from pathlib import Path

from engine.football_data import import_football_data_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Folder containing downloaded Football-Data CSV files")
    parser.add_argument("--output", default="data/raw/matches.csv", help="Combined normalized CSV output path")
    args = parser.parse_args(argv)

    matches = import_football_data_directory(args.input_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output, index=False)
    print(f"Imported {len(matches)} completed matches from {args.input_dir}")
    print(f"Competitions: {matches['competition'].nunique()}")
    print(f"Seasons: {matches['season'].nunique()}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
