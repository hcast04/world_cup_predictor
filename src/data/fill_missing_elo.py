from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW, load_elo_ratings, load_teams


CONFEDERATION_ELO_DEFAULTS = {
    "UEFA": 1750,
    "CONMEBOL": 1780,
    "CAF": 1650,
    "AFC": 1600,
    "CONCACAF": 1580,
    "OFC": 1450,
}


def fill_missing_elo(output_path: Path | None = None) -> pd.DataFrame:
    """
    Fill missing Elo ratings for teams in teams_2026.csv.

    Existing Elo values are kept.
    Missing Elo values are filled using rough confederation-aware defaults.

    Output columns:
    - rank
    - team
    - elo
    - date
    - elo_source

    This is temporary. Later, replace estimated Elo values with real ratings.
    """
    output_path = output_path or DATA_RAW / "elo_ratings_current.csv"

    teams = load_teams()
    elo = load_elo_ratings()

    # Existing Elo rows are treated as manually verified or seeded.
    existing = elo.copy()
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
    print(f"\nSaved filled Elo ratings to {DATA_RAW / 'elo_ratings_current.csv'}")