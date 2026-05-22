import pandas as pd

from src.data.loaders import DATA_RAW


def main() -> None:
    path = DATA_RAW / "club_player_stats_2025_2026.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Save your downloaded club player stats file there first."
        )

    df = pd.read_csv(path)

    print("\nClub player data inspection")
    print("---------------------------")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()