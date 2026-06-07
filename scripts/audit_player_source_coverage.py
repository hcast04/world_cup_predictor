import pandas as pd

from src.data.loaders import DATA_PROCESSED


def main() -> None:
    path = DATA_PROCESSED / "club_player_stats_normalized.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m src.data.normalize_club_player_sources first."
        )

    df = pd.read_csv(path)

    print("\nPlayer source coverage")
    print("----------------------")
    print(f"Rows: {len(df):,}")
    print(f"Teams: {df['team'].nunique():,}")
    print(f"Clubs: {df['club'].nunique():,}")
    print(f"Leagues: {df['league'].nunique():,}")
    print(f"Source files: {df['source_file'].nunique():,}")

    print("\nRows by source file:")
    print(df["source_file"].value_counts().to_string())

    print("\nRows by league:")
    print(df["league"].value_counts().head(50).to_string())

    print("\nRows by World Cup team/nation:")
    print(df["team"].value_counts().head(80).to_string())


if __name__ == "__main__":
    main()