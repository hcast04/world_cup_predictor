import pandas as pd

from src.data.loaders import DATA_PROCESSED


def main() -> None:
    path = DATA_PROCESSED / "historical_matches.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m src.data.build_historical_matches first."
        )

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    print("\nHistorical match summary")
    print("------------------------")
    print(f"Rows: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Teams: {len(set(df['team_a']).union(set(df['team_b']))):,}")
    print(f"Tournaments: {df['tournament'].nunique():,}")

    print("\nResult distribution:")
    print(df["result"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

    print("\nAverage goals:")
    print(f"Team A goals: {df['goals_a'].mean():.2f}")
    print(f"Team B goals: {df['goals_b'].mean():.2f}")
    print(f"Total goals: {(df['goals_a'] + df['goals_b']).mean():.2f}")

    print("\nMost common tournaments since 2010:")
    print(df["tournament"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()