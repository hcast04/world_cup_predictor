from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW
from src.utils.names import load_team_name_map, normalize_team_column


def load_groups(path: Path | None = None) -> pd.DataFrame:
    """
    Load the 2026 World Cup group assignments.

    Expected columns:
    - group
    - slot
    - team

    The slot column is important because the official fixture pattern uses
    group positions such as A1 vs A2, A3 vs A4, etc.
    """
    path = path or DATA_RAW / "groups_2026.csv"
    df = pd.read_csv(path)

    required_columns = {"group", "slot", "team"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in groups file: {missing}")

    df["group"] = df["group"].astype(str).str.strip()
    df["slot"] = pd.to_numeric(df["slot"], errors="raise").astype(int)
    df["team"] = df["team"].astype(str).str.strip()

    name_map = load_team_name_map()
    df = normalize_team_column(df, "team", name_map)

    duplicate_slots = df.duplicated(subset=["group", "slot"], keep=False)
    if duplicate_slots.any():
        raise ValueError(
            "Duplicate group/slot assignments found:\n"
            f"{df.loc[duplicate_slots].to_string(index=False)}"
        )

    return df