from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW, PROJECT_ROOT, load_teams


DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


CONFEDERATION_ELO_DEFAULTS = {
    "UEFA": 1750,
    "CONMEBOL": 1780,
    "CAF": 1650,
    "AFC": 1600,
    "CONCACAF": 1580,
    "OFC": 1450,
}


def load_seed_elo(path: Path | None = None) -> pd.DataFrame:
    """
    Load manually seeded Elo ratings.

    Expected columns:
    - rank
    - team
    - elo
    - date
    """
    path = path or DATA_RAW / "elo_ratings_seed.csv"
    df = pd.read_csv(path)

    required_columns = {"rank", "team", "elo", "date"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in seed Elo file: {missing}")

    df["team"] = df["team"].astype(str).str.strip()
    df["elo"] = pd.to_numeric(df["elo"], errors="raise")
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    return df


def fill_missing_elo(output_path: Path | None = None) -> pd.DataFrame:
    """
    Create complete model-ready Elo ratings for all teams in teams_2026.csv.

    Existing seed Elo values are kept.
    Missing values are filled using temporary confederation-aware defaults.

    Output columns:
    - rank
    - team
    - elo
    - date
    - elo_source
    """
    output_path = output_path or DATA_PROCESSED / "elo_ratings_model.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    teams = load_teams()
    seed_elo = load_seed_elo()

    existing = seed_elo.copy()
    existing["elo_source"] = "seed_or_manual"

    existing_teams = set(existing["team"])
    missing_teams = teams[~teams["team"].isin(existing_teams)].copy()

    rows = []

    for _, row in missing_teams.iterrows():
        team = row["team"]
        confederation = row["confederation"]

        if confederation not in CONFEDERATION_ELO_DEFAULTS:
            raise ValueError(f"Missing Elo default for confederation: {confederation}")

        rows.append(
            {
                "rank": 999,
                "team": team,
                "elo": CONFEDERATION_ELO_DEFAULTS[confederation],
                "date": existing["date"].max(),
                "elo_source": "confederation_default",
            }
        )

    filled = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)

    filled = filled[filled["team"].isin(set(teams["team"]))].copy()

    filled = filled.sort_values(
        by=["elo", "team"],
        ascending=[False, True],
    ).reset_index(drop=True)

    filled["rank"] = range(1, len(filled) + 1)

    filled.to_csv(output_path, index=False)

    return filled


if __name__ == "__main__":
    filled = fill_missing_elo()
    print(filled.to_string(index=False))
    print(f"\nSaved model Elo ratings to {DATA_PROCESSED / 'elo_ratings_model.csv'}")