"""Import manually downloaded Football-Data.co.uk season CSV files.

This importer reads local files only. It does not scrape or download from the
provider. Keep the provider's original CSVs in data/raw/football-data/.
"""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from engine.data import MatchDataError, validate_matches

DIVISION_NAMES = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "E2": "England League One",
    "E3": "England League Two",
    "EC": "England National League",
    "SC0": "Scotland Premiership",
    "SC1": "Scotland Championship",
    "SC2": "Scotland League One",
    "SC3": "Scotland League Two",
    "D1": "Germany Bundesliga",
    "D2": "Germany 2. Bundesliga",
    "I1": "Italy Serie A",
    "I2": "Italy Serie B",
    "SP1": "Spain La Liga",
    "SP2": "Spain Segunda Division",
    "F1": "France Ligue 1",
    "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie",
    "B1": "Belgium First Division A",
    "P1": "Portugal Primeira Liga",
    "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
}


def _season_from_path(path: Path) -> str:
    """Find a four-digit season code such as 1011 in path or filename."""
    candidates = [path.stem, *[part for part in path.parts[-3:-1]]]
    for candidate in candidates:
        match = re.search(r"(?<!\d)(\d{4})(?!\d)", candidate)
        if match:
            code = match.group(1)
            start = int(code[:2])
            end = int(code[2:])
            start_year = 1900 + start if start >= 90 else 2000 + start
            end_year = 1900 + end if end >= 90 else 2000 + end
            if end_year == start_year + 1:
                return f"{start_year}-{end_year}"
    raise MatchDataError(
        f"cannot infer season from {path}; include a season code like 1011 "
        "in the filename or parent folder"
    )


def import_football_data_file(path: str | Path) -> pd.DataFrame:
    """Convert one locally downloaded Football-Data CSV to engine schema.

    Supports common file layouts such as data/raw/football-data/1011/E0.csv
    and data/raw/football-data/E0_1011.csv. Unplayed fixtures are excluded.
    All original source columns are retained for later odds/statistics work.
    """
    source_path = Path(path)
    try:
        raw = pd.read_csv(source_path, encoding="cp1252")
    except Exception as exc:
        raise MatchDataError(f"could not read {source_path}: {exc}") from exc

    raw.columns = [str(column).replace("\ufeff", "").strip() for column in raw.columns]
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(raw.columns)
    if missing:
        raise MatchDataError(
            f"{source_path.name} missing Football-Data columns: "
            f"{', '.join(sorted(missing))}"
        )

    division = None
    if "Div" in raw.columns and raw["Div"].dropna().nunique() == 1:
        division = str(raw["Div"].dropna().iloc[0]).strip()
    if not division:
        match = re.search(r"(?<![A-Z])(SC[0-3]|E[0-3]|EC|D[12]|I[12]|SP[12]|F[12]|N1|B1|P1|T1|G1)(?![A-Z])", source_path.stem.upper())
        if match:
            division = match.group(1)
    if not division:
        raise MatchDataError(
            f"cannot determine league code for {source_path.name}; "
            "keep the Div column or include the code in the filename"
        )

    season = _season_from_path(source_path)
    result = raw.copy()
    result["date"] = pd.to_datetime(result["Date"], dayfirst=True, errors="coerce")
    if result["date"].isna().any():
        bad_count = int(result["date"].isna().sum())
        raise MatchDataError(f"{source_path.name} has {bad_count} invalid match dates")

    # Current-season files can contain scheduled fixtures with no final score.
    home_goals = pd.to_numeric(result["FTHG"], errors="coerce")
    away_goals = pd.to_numeric(result["FTAG"], errors="coerce")
    both_missing = home_goals.isna() & away_goals.isna()
    one_missing = home_goals.isna() ^ away_goals.isna()
    if one_missing.any():
        raise MatchDataError(f"{source_path.name} has rows with only one full-time score")
    result = result.loc[~both_missing].copy()
    result["home_goals"] = home_goals.loc[~both_missing]
    result["away_goals"] = away_goals.loc[~both_missing]
    result["competition"] = DIVISION_NAMES.get(division, f"Football-Data {division}")
    result["season"] = season
    result["home_team"] = result["HomeTeam"].astype("string").str.strip()
    result["away_team"] = result["AwayTeam"].astype("string").str.strip()
    result["source"] = "Football-Data.co.uk"
    result["source_file"] = source_path.name

    # Normalize date to a timezone-aware value in the shared engine schema.
    result["date"] = pd.to_datetime(result["date"], utc=True)
    return validate_matches(result)


def import_football_data_directory(directory: str | Path) -> pd.DataFrame:
    """Import all CSV files in a directory tree, retaining per-file metadata."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise MatchDataError(f"no CSV files found under {root}")
    frames = [import_football_data_file(file) for file in files]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return validate_matches(combined)
