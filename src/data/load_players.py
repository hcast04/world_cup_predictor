from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW
from src.utils.names import load_team_name_map, normalize_team_column


def load_players(path: Path | None = None) -> pd.DataFrame:
    """
    Load player-level inputs for the Golden Boot model.

    Expected columns:
    - player
    - team
    - position
    - expected_minutes_per_match
    - starter_probability
    - goals_per90
    - xg_per90
    - is_penalty_taker
    - scoring_weight_source
    """
    path = path or DATA_RAW / "players_2026_seed.csv"
    df = pd.read_csv(path)

    required_columns = {
        "player",
        "team",
        "position",
        "expected_minutes_per_match",
        "starter_probability",
        "goals_per90",
        "xg_per90",
        "is_penalty_taker",
        "scoring_weight_source",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in players file: {missing}")

    df["player"] = df["player"].astype(str).str.strip()
    df["team"] = df["team"].astype(str).str.strip()
    df["position"] = df["position"].astype(str).str.strip()

    name_map = load_team_name_map()
    df = normalize_team_column(df, "team", name_map)

    numeric_columns = [
        "expected_minutes_per_match",
        "starter_probability",
        "goals_per90",
        "xg_per90",
        "is_penalty_taker",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df