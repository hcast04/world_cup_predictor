import pandas as pd

from src.data.loaders import DATA_RAW


def validate_file(path, required_columns):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    return df


def main() -> None:
    results_path = DATA_RAW / "historical_results.csv"
    goalscorers_path = DATA_RAW / "historical_goalscorers.csv"
    shootouts_path = DATA_RAW / "historical_shootouts.csv"

    results = validate_file(
        results_path,
        {
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "city",
            "country",
            "neutral",
        },
    )

    goalscorers = validate_file(
        goalscorers_path,
        {
            "date",
            "home_team",
            "away_team",
            "team",
            "scorer",
            "own_goal",
            "penalty",
        },
    )

    shootouts = validate_file(
        shootouts_path,
        {
            "date",
            "home_team",
            "away_team",
            "winner",
        },
    )

    print("\nHistorical data validation")
    print("--------------------------")
    print(f"Historical results: {len(results):,} rows")
    print(f"Historical goal scorers: {len(goalscorers):,} rows")
    print(f"Historical shootouts: {len(shootouts):,} rows")

    print("\nDate ranges")
    print(f"Results: {results['date'].min()} to {results['date'].max()}")
    print(f"Goalscorers: {goalscorers['date'].min()} to {goalscorers['date'].max()}")
    print(f"Shootouts: {shootouts['date'].min()} to {shootouts['date'].max()}")

    print("\nRecent tournaments in results:")
    recent = results.sort_values("date", ascending=False).head(200)
    print(sorted(recent["tournament"].dropna().unique())[:50])

    print("\nValidation passed.")


if __name__ == "__main__":
    main()