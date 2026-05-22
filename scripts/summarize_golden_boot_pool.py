import pandas as pd

from src.data.loaders import DATA_PROCESSED


def main() -> None:
    path = DATA_PROCESSED / "golden_boot_player_pool.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m src.data.build_golden_boot_pool first."
        )

    df = pd.read_csv(path)

    print("\nGolden Boot player pool summary")
    print("-------------------------------")
    print(f"Rows: {len(df):,}")
    print(f"Teams represented: {df['team'].nunique():,}")
    print(f"Players per team: min={df.groupby('team').size().min()}, max={df.groupby('team').size().max()}")

    print("\nTop 40 candidates by club scoring score:")
    print(
        df.sort_values("club_scoring_score_shrunk", ascending=False)[
            [
                "player",
                "team",
                "position",
                "club",
                "league",
                "minutes",
                "goals",
                "xg",
                "xg_per90",
                "goals_per90",
                "expected_minutes_per_match",
                "starter_probability",
                "is_penalty_taker",
                "club_scoring_score_shrunk",
            ]
        ]
        .head(40)
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\nTeams with fewest candidates:")
    print(df.groupby("team").size().sort_values().head(20).to_string())


if __name__ == "__main__":
    main()