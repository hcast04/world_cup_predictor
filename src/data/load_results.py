from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW
from src.utils.names import load_team_name_map, normalize_team_column


def load_manual_match_results(path: Path | None = None) -> pd.DataFrame:
    """
    Load manually entered match results.

    Expected columns:
    - match_id
    - stage
    - group
    - date
    - team_a
    - team_b
    - goals_a
    - goals_b
    - status

    This file should contain only matches whose results are known.
    """
    path = path or DATA_RAW / "manual_match_results.csv"
    df = pd.read_csv(path)

    required_columns = {
        "match_id",
        "stage",
        "group",
        "date",
        "team_a",
        "team_b",
        "goals_a",
        "goals_b",
        "status",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in manual match results file: {missing}")

    if df.empty:
        return df

    text_columns = ["match_id", "stage", "group", "team_a", "team_b", "status"]
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["goals_a"] = pd.to_numeric(df["goals_a"], errors="raise").astype(int)
    df["goals_b"] = pd.to_numeric(df["goals_b"], errors="raise").astype(int)

    name_map = load_team_name_map()
    df = normalize_team_column(df, "team_a", name_map)
    df = normalize_team_column(df, "team_b", name_map)

    return df