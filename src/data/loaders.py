from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def load_teams(path: Path | None = None) -> pd.DataFrame:
    """
    Load the 2026 World Cup team metadata.

    Expected columns:
    - team
    - confederation
    - is_host
    - group
    """
    path = path or DATA_RAW / "teams_2026.csv"
    df = pd.read_csv(path)

    required_columns = {"team", "confederation", "is_host", "group"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in teams file: {missing}")

    df["team"] = df["team"].astype(str).str.strip()
    df["confederation"] = df["confederation"].astype(str).str.strip()
    df["is_host"] = df["is_host"].astype(int)
    df["group"] = df["group"].astype(str).str.strip()

    return df

def load_elo_ratings(path: Path | None = None) -> pd.DataFrame:
    """
    Load current Elo ratings.

    Expected columns:
    - team
    - elo
    - date
    """
    path = path or DATA_RAW / "elo_ratings_current.csv"
    df = pd.read_csv(path)

    required_columns = {"team", "elo", "date"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in Elo ratings file: {missing}")

    df["team"] = df["team"].astype(str).str.strip()
    df["elo"] = pd.to_numeric(df["elo"], errors="raise")
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    return df


def load_fixtures(path: Path | None = None) -> pd.DataFrame:
    """
    Load World Cup 2026 fixtures.

    Expected columns:
    - match_id
    - stage
    - group
    - date
    - team_a
    - team_b
    - venue
    - city
    - country
    """
    path = path or DATA_RAW / "fixtures_2026.csv"
    df = pd.read_csv(path)

    required_columns = {
        "match_id",
        "stage",
        "group",
        "date",
        "team_a",
        "team_b",
        "venue",
        "city",
        "country",
    }
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in fixtures file: {missing}")

    text_columns = ["stage", "group", "team_a", "team_b", "venue", "city", "country"]
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()

    df["date"] = pd.to_datetime(df["date"], errors="raise")

    return df


def load_baseline_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for loading all baseline raw files.
    """
    teams = load_teams()
    elo = load_elo_ratings()
    fixtures = load_fixtures()

    return teams, elo, fixtures