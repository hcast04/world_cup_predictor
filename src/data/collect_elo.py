from pathlib import Path
import re
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.data.loaders import DATA_RAW
from src.utils.names import load_team_name_map, normalize_team_name


ELO_URL = "https://www.eloratings.net/"


def _parse_rating_date(page_text: str) -> str:
    """
    Parse the rating date from eloratings.net text.

    If the date cannot be parsed, use today's date as the collection date.
    """
    match = re.search(
        r"Ratings and Statistics as of\s+([A-Za-z]{3}\s+[A-Za-z]{3}\s+\d{1,2}\s+\d{4})",
        page_text,
    )

    if match is None:
        return date.today().isoformat()

    raw_date = match.group(1)
    return pd.to_datetime(raw_date).date().isoformat()


def collect_current_elo(output_path: Path | None = None) -> pd.DataFrame:
    """
    Collect current World Football Elo ratings from eloratings.net.

    Output columns:
    - rank
    - team
    - elo
    - date
    """
    output_path = output_path or DATA_RAW / "elo_ratings_current.csv"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        )
    }

    response = requests.get(ELO_URL, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    text = soup.get_text(" ", strip=True)

    rating_date = _parse_rating_date(text)
    name_map = load_team_name_map()

    # Pattern examples:
    # 1. Spain. 2165
    # 2. Argentina. 2113
    #
    # The team group is deliberately non-greedy and rating must be 3 or 4 digits.
    pattern = re.compile(r"(\d{1,3})\.\s+(.+?)\.\s+(\d{3,4})(?=\s+\d{1,3}\.|$)")

    rows = []

    for match in pattern.finditer(text):
        rank = int(match.group(1))
        raw_team = match.group(2).strip()
        elo = int(match.group(3))

        team = normalize_team_name(raw_team, name_map)

        rows.append(
            {
                "rank": rank,
                "team": team,
                "elo": elo,
                "date": rating_date,
            }
        )

    if not rows:
        # Save the received text to help debug future website changes.
        debug_path = DATA_RAW / "eloratings_debug_text.txt"
        debug_path.write_text(text, encoding="utf-8")
        raise ValueError(
            "No Elo rows parsed. Saved received page text to "
            f"{debug_path}. Open it to inspect the website structure."
        )

    df = pd.DataFrame(rows)

    df = (
        df.sort_values("rank")
        .drop_duplicates(subset=["team"], keep="first")
        .reset_index(drop=True)
    )

    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    elo = collect_current_elo()
    print(elo.head(20).to_string(index=False))
    print(f"\nSaved {len(elo)} Elo ratings to {DATA_RAW / 'elo_ratings_current.csv'}")