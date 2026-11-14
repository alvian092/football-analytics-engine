"""Historical match CSV loading and validation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "date", "competition", "season", "home_team", "away_team",
    "home_goals", "away_goals",
}
OPTIONAL_NUMERIC_COLUMNS = {"home_xg", "away_xg"}


class MatchDataError(ValueError):
    """Raised when historical match data is incomplete or inconsistent."""


def load_matches(path: str | Path) -> pd.DataFrame:
    """Load and validate a match CSV; dates are normalized to pandas timestamps."""
    frame = pd.read_csv(path)
    return validate_matches(frame)


def validate_matches(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of match data or raise MatchDataError.

    xG columns are optional. When supplied, blank xG values are allowed, but
    non-blank values must be finite and non-negative.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise MatchDataError(f"missing required columns: {', '.join(sorted(missing))}")

    result = frame.copy()
    for column in ("competition", "season", "home_team", "away_team"):
        result[column] = result[column].astype("string").str.strip()
        if result[column].isna().any() or result[column].eq("").any():
            raise MatchDataError(f"{column} contains blank values")

    result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=True)
    if result["date"].isna().any():
        raise MatchDataError("date contains invalid or blank values")

    for column in ("home_goals", "away_goals"):
        numeric = pd.to_numeric(result[column], errors="coerce")
        if numeric.isna().any() or (numeric < 0).any():
            raise MatchDataError(f"{column} must contain non-negative integers")
        if ((numeric % 1) != 0).any():
            raise MatchDataError(f"{column} must contain whole numbers")
        result[column] = numeric.astype("int64")

    for column in OPTIONAL_NUMERIC_COLUMNS.intersection(result.columns):
        numeric = pd.to_numeric(result[column], errors="coerce")
        invalid_text = result[column].notna() & numeric.isna()
        if invalid_text.any() or (numeric.dropna() < 0).any():
            raise MatchDataError(f"{column} must be blank or a non-negative number")
        result[column] = numeric

    if (result["home_team"] == result["away_team"]).any():
        raise MatchDataError("home_team and away_team cannot be the same")

    duplicate_key = ["date", "competition", "season", "home_team", "away_team"]
    if result.duplicated(duplicate_key).any():
        raise MatchDataError("duplicate match rows found")

    return result.sort_values("date").reset_index(drop=True)
