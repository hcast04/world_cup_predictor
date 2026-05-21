import pandas as pd

from src.data.loaders import DATA_PROCESSED


def main() -> None:
    path = DATA_PROCESSED / "match_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m src.data.build_match_features first."
        )

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    print("\nMatch feature summary")
    print("---------------------")
    print(f"Rows: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Columns: {len(df.columns):,}")

    feature_cols = [
        col
        for col in df.columns
        if "last_5" in col or "last_10" in col or col.endswith("_diff")
    ]

    missing_rates = df[feature_cols].isna().mean().sort_values(ascending=False)

    print("\nHighest missing-rate feature columns:")
    print(missing_rates.head(20).to_string())

    print("\nSample feature columns:")
    sample_cols = [
        "date",
        "team_a",
        "team_b",
        "goals_a",
        "goals_b",
        "result",
        "team_a_points_last_10",
        "team_b_points_last_10",
        "points_diff_last_10",
        "team_a_goal_diff_last_10",
        "team_b_goal_diff_last_10",
        "goal_diff_diff_last_10",
    ]
    sample_cols = [col for col in sample_cols if col in df.columns]

    print(df[sample_cols].tail(20).to_string(index=False))


if __name__ == "__main__":
    main()