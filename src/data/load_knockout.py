from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW
from src.utils.names import load_team_name_map, normalize_team_column


def load_manual_knockout_fixtures(path: Path | None = None) -> pd.DataFrame:
    """
    Load manually entered knockout fixtures.

    Expected columns:
    - match_id
    - stage
    - date
    - team_a
    - team_b
    - venue
    - city
    - country
    - status

    This file is intentionally manual because after the group stage you will know
    the actual knockout fixtures and can enter them directly.
    """
    path = path or DATA_RAW / "manual_knockout_fixtures.csv"
    df = pd.read_csv(path)

    required_columns = {
        "match_id",
        "stage",
        "date",
        "team_a",
        "team_b",
        "venue",
        "city",
        "country",
        "status",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in knockout fixtures file: {missing}")

    if df.empty:
        return df

    text_columns = ["match_id", "stage", "team_a", "team_b", "venue", "city", "country", "status"]
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    name_map = load_team_name_map()
    df = normalize_team_column(df, "team_a", name_map)
    df = normalize_team_column(df, "team_b", name_map)

    return df