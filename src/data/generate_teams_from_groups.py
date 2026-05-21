from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW
from src.data.load_groups import load_groups


CONFEDERATION_MAP = {
    "Mexico": "CONCACAF",
    "South Africa": "CAF",
    "South Korea": "AFC",
    "Czechia": "UEFA",
    "Canada": "CONCACAF",
    "Bosnia and Herzegovina": "UEFA",
    "Qatar": "AFC",
    "Switzerland": "UEFA",
    "Brazil": "CONMEBOL",
    "Morocco": "CAF",
    "Haiti": "CONCACAF",
    "Scotland": "UEFA",
    "United States": "CONCACAF",
    "Paraguay": "CONMEBOL",
    "Australia": "AFC",
    "Turkey": "UEFA",
    "Germany": "UEFA",
    "Curacao": "CONCACAF",
    "Ivory Coast": "CAF",
    "Ecuador": "CONMEBOL",
    "Netherlands": "UEFA",
    "Japan": "AFC",
    "Sweden": "UEFA",
    "Tunisia": "CAF",
    "Belgium": "UEFA",
    "Egypt": "CAF",
    "Iran": "AFC",
    "New Zealand": "OFC",
    "Spain": "UEFA",
    "Cape Verde": "CAF",
    "Saudi Arabia": "AFC",
    "Uruguay": "CONMEBOL",
    "France": "UEFA",
    "Senegal": "CAF",
    "Iraq": "AFC",
    "Norway": "UEFA",
    "Argentina": "CONMEBOL",
    "Algeria": "CAF",
    "Austria": "UEFA",
    "Jordan": "AFC",
    "Portugal": "UEFA",
    "DR Congo": "CAF",
    "Uzbekistan": "AFC",
    "Colombia": "CONMEBOL",
    "England": "UEFA",
    "Croatia": "UEFA",
    "Ghana": "CAF",
    "Panama": "CONCACAF",
}

HOSTS = {"Canada", "Mexico", "United States"}


def generate_teams_from_groups(output_path: Path | None = None) -> pd.DataFrame:
    output_path = output_path or DATA_RAW / "teams_2026.csv"

    groups = load_groups()

    rows = []

    for _, row in groups.iterrows():
        team = row["team"]
        group = row["group"]

        if team == "TBD":
            continue

        if team not in CONFEDERATION_MAP:
            raise ValueError(f"Missing confederation for team: {team}")

        rows.append(
            {
                "team": team,
                "confederation": CONFEDERATION_MAP[team],
                "is_host": 1 if team in HOSTS else 0,
                "group": group,
            }
        )

    teams = pd.DataFrame(rows).sort_values(["group", "team"]).reset_index(drop=True)
    teams.to_csv(output_path, index=False)

    return teams


if __name__ == "__main__":
    teams = generate_teams_from_groups()
    print(teams.to_string(index=False))
    print(f"\nSaved teams to {DATA_RAW / 'teams_2026.csv'}")