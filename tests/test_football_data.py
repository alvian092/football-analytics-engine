from pathlib import Path

import pandas as pd
import pytest

from engine.data import MatchDataError
from engine.football_data import import_football_data_file, import_football_data_directory


def test_imports_football_data_csv_and_maps_columns(tmp_path: Path):
    path = tmp_path / "E0_1011.csv"
    pd.DataFrame([
        {
            "Div": "E0", "Date": "14/08/10", "HomeTeam": "A", "AwayTeam": "B",
            "FTHG": 2, "FTAG": 1, "HTHG": 1, "HTAG": 0, "B365H": 1.8,
        },
        {
            "Div": "E0", "Date": "15/08/10", "HomeTeam": "C", "AwayTeam": "D",
            "FTHG": None, "FTAG": None, "HTHG": None, "HTAG": None, "B365H": None,
        },
    ]).to_csv(path, index=False, encoding="cp1252")

    result = import_football_data_file(path)
    assert len(result) == 1
    assert result.loc[0, "competition"] == "England Premier League"
    assert result.loc[0, "season"] == "2010-2011"
    assert result.loc[0, "home_goals"] == 2
    assert result.loc[0, "B365H"] == 1.8
    assert result.loc[0, "source"] == "Football-Data.co.uk"


def test_import_directory_combines_files(tmp_path: Path):
    for code in ("E0", "E1"):
        pd.DataFrame([{
            "Div": code, "Date": "14/08/10", "HomeTeam": "A", "AwayTeam": "B",
            "FTHG": 2, "FTAG": 1,
        }]).to_csv(tmp_path / f"{code}_1011.csv", index=False)
    result = import_football_data_directory(tmp_path)
    assert len(result) == 2
    assert set(result["competition"]) == {
        "England Premier League", "England Championship"
    }


def test_one_missing_full_time_score_is_rejected(tmp_path: Path):
    path = tmp_path / "E0_1011.csv"
    pd.DataFrame([{
        "Div": "E0", "Date": "14/08/10", "HomeTeam": "A", "AwayTeam": "B",
        "FTHG": 2, "FTAG": None,
    }]).to_csv(path, index=False)
    with pytest.raises(MatchDataError, match="only one"):
        import_football_data_file(path)
