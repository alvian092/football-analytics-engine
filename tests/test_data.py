import pandas as pd
import pytest

from engine.data import MatchDataError, validate_matches


def sample():
    return pd.DataFrame([{
        "date": "2025-08-15",
        "competition": "Premier League",
        "season": "2025-2026",
        "home_team": " Liverpool ",
        "away_team": "Bournemouth",
        "home_goals": 4,
        "away_goals": 2,
        "home_xg": 2.4,
        "away_xg": 0.8,
        "source": "example",
    }])


def test_valid_data_is_cleaned():
    result = validate_matches(sample())
    assert result.loc[0, "home_team"] == "Liverpool"
    assert result.loc[0, "home_goals"] == 4
    assert str(result.loc[0, "date"].tz) == "UTC"


def test_missing_required_column_fails():
    with pytest.raises(MatchDataError, match="missing required"):
        validate_matches(sample().drop(columns=["home_goals"]))


def test_negative_goals_fail():
    frame = sample()
    frame.loc[0, "home_goals"] = -1
    with pytest.raises(MatchDataError, match="non-negative"):
        validate_matches(frame)


def test_fractional_goals_fail():
    frame = sample()
    frame.loc[0, "home_goals"] = 1.5
    with pytest.raises(MatchDataError, match="whole numbers"):
        validate_matches(frame)


def test_invalid_xg_fails():
    frame = sample()
    frame.loc[0, "home_xg"] = -0.2
    with pytest.raises(MatchDataError, match="home_xg"):
        validate_matches(frame)


def test_duplicate_match_fails():
    frame = pd.concat([sample(), sample()], ignore_index=True)
    with pytest.raises(MatchDataError, match="duplicate"):
        validate_matches(frame)
