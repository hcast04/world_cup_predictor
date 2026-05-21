from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW


def load_team_name_map(path: Path | None = None) -> dict[str, str]:
    """
    Load source-to-canonical team name mappings.

    Expected columns:
    - source_name
    - canonical_name
    - source
    """
    path = path or DATA_RAW / "team_name_map.csv"
    df = pd.read_csv(path)

    required_columns = {"source_name", "canonical_name", "source"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in team name map: {missing}")

    df["source_name"] = df["source_name"].astype(str).str.strip()
    df["canonical_name"] = df["canonical_name"].astype(str).str.strip()

    return dict(zip(df["source_name"], df["canonical_name"]))


def normalize_team_name(name: str, name_map: dict[str, str] | None = None) -> str:
    """
    Convert a team name from any source into the project's canonical name.

    If the name is not in the map, return the stripped input unchanged.
    """
    clean_name = str(name).strip()

    if name_map is None:
        name_map = load_team_name_map()

    return name_map.get(clean_name, clean_name)


def normalize_team_column(
    df: pd.DataFrame,
    column: str,
    name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Normalize one team-name column in a DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    df = df.copy()
    name_map = name_map or load_team_name_map()

    df[column] = df[column].map(lambda x: normalize_team_name(x, name_map))

    return df