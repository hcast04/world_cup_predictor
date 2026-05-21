from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW, DATA_PROCESSED
from src.utils.names import load_team_name_map, normalize_team_column


def build_historical_matches(
    input_path: Path | None = None,
    output_path: Path | None = None,
    start_date: str = "2010-01-01",
) -> pd.DataFrame:
    """
    Build a model-ready historical match dataset.

    The raw data uses home/away. We convert this into team_a/team_b format.
    Rows without final scores are dropped.
    """
    input_path = input_path or DATA_RAW / "historical_results.csv"
    output_path = output_path or DATA_PROCESSED / "historical_matches.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df[df["date"] >= pd.to_datetime(start_date)].copy()

    df = df.rename(
        columns={
            "home_team": "team_a",
            "away_team": "team_b",
            "home_score": "goals_a",
            "away_score": "goals_b",
        }
    )

    before_drop = len(df)

    df["goals_a"] = pd.to_numeric(df["goals_a"], errors="coerce")
    df["goals_b"] = pd.to_numeric(df["goals_b"], errors="coerce")

    df = df.dropna(subset=["goals_a", "goals_b"]).copy()

    dropped_rows = before_drop - len(df)

    name_map = load_team_name_map()
    df = normalize_team_column(df, "team_a", name_map)
    df = normalize_team_column(df, "team_b", name_map)

    df["goals_a"] = df["goals_a"].astype(int)
    df["goals_b"] = df["goals_b"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)

    df["result"] = "draw"
    df.loc[df["goals_a"] > df["goals_b"], "result"] = "team_a_win"
    df.loc[df["goals_b"] > df["goals_a"], "result"] = "team_b_win"

    keep_columns = [
        "date",
        "team_a",
        "team_b",
        "goals_a",
        "goals_b",
        "result",
        "tournament",
        "city",
        "country",
        "neutral",
    ]

    out = df[keep_columns].sort_values("date").reset_index(drop=True)
    out.to_csv(output_path, index=False)

    print(f"Dropped {dropped_rows:,} rows without complete scores.")

    return out


if __name__ == "__main__":
    matches = build_historical_matches()
    print(matches.head().to_string(index=False))
    print()
    print(matches.tail().to_string(index=False))
    print(f"\nSaved {len(matches):,} rows to {DATA_PROCESSED / 'historical_matches.csv'}")