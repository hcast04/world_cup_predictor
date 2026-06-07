import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.data.loaders import DATA_RAW


URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"

TEAM_NAME_MAP = {
    "Czech Republic": "Czechia",
    "Curaçao": "Curacao",
    "United States of America": "United States",
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
}

def _clean_text(value: str) -> str:
    value = str(value)
    value = re.sub(r"\[[^\]]*\]", "", value)
    value = value.replace("\xa0", " ")
    return value.strip()


def _normalize_position(pos: str) -> str:
    pos = _clean_text(pos).upper()

    mapping = {
        "GK": "GK",
        "DF": "DF",
        "MF": "MF",
        "FW": "FW",
    }

    return mapping.get(pos, pos)


def collect_world_cup_squads_wikipedia(
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Collect currently available 2026 World Cup squad tables from Wikipedia.

    This is a practical first-pass collector. The output should be reviewed,
    because some teams may not yet have final squads listed.
    """
    output_path = output_path or DATA_RAW / "world_cup_squads_2026.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        URL,
        headers={"User-Agent": "world-cup-predictor/0.1"},
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    rows = []

    current_team = None

    for element in soup.find_all(["h3", "table"]):
        if element.name == "h3":
            headline = element.get_text(" ", strip=True)
            headline = _clean_text(headline)

            # h3 usually contains team names on the squad page.
            if headline and headline not in {"Notes", "References", "External links"}:
                current_team = TEAM_NAME_MAP.get(headline, headline)

        elif element.name == "table" and current_team:
            table_classes = element.get("class", [])

            if "wikitable" not in table_classes:
                continue

            try:
                table = pd.read_html(str(element))[0]
            except ValueError:
                continue

            table.columns = [str(col).strip() for col in table.columns]

            # Typical squad table has columns like:
            # No., Pos., Player, Date of birth (age), Caps, Goals, Club
            player_col = next((c for c in table.columns if "Player" in c), None)
            pos_col = next((c for c in table.columns if "Pos" in c), None)
            club_col = next((c for c in table.columns if "Club" in c), None)

            if player_col is None or pos_col is None:
                continue

            for _, row in table.iterrows():
                player = _clean_text(row.get(player_col, ""))

                if not player or player.lower() == "nan":
                    continue

                position = _normalize_position(row.get(pos_col, ""))
                club = _clean_text(row.get(club_col, "")) if club_col else ""

                rows.append(
                    {
                        "team": current_team,
                        "player": player,
                        "position": position,
                        "club": club,
                        "source": "wikipedia_2026_world_cup_squads",
                        "notes": "",
                    }
                )

    out = pd.DataFrame(rows)

    if out.empty:
        print("No squad rows found. The page structure may have changed or squads may not be listed yet.")
        out = pd.DataFrame(
            columns=["team", "player", "position", "club", "source", "notes"]
        )

    out = out.drop_duplicates(subset=["team", "player"]).sort_values(
        ["team", "position", "player"]
    )

    out.to_csv(output_path, index=False)

    print("\nWorld Cup squad collection")
    print("--------------------------")
    print(f"Rows: {len(out):,}")
    if not out.empty:
        print(f"Teams: {out['team'].nunique():,}")
        print("\nRows by team:")
        print(out["team"].value_counts().sort_index().to_string())

    print(f"\nSaved to {output_path}")

    return out


if __name__ == "__main__":
    collect_world_cup_squads_wikipedia()