import pandas as pd

from src.data.loaders import DATA_PROCESSED


def main() -> None:
    path = DATA_PROCESSED / "player_scoring_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m src.data.build_player_scoring_features first."
        )

    df = pd.read_csv(path)

    print("\nPlayer scoring feature summary")
    print("------------------------------")
    print(f"Rows: {len(df):,}")
    print(f"Teams: {df['team'].nunique():,}")
    print(f"Clubs: {df['club'].nunique():,}")
    print(f"Leagues: {df['league'].nunique():,}")

    print("\nTop 30 players by shrunk club scoring score:")
    print(
        df[
            [
                "player",
                "team",
                "position",
                "club",
                "league",
                "minutes",
                "goals",
                "xg",
                "npxg",
                "goals_per90",
                "xg_per90",
                "npxg_per90",
                "club_scoring_score_shrunk",
                "is_penalty_taker_proxy",
            ]
        ]
        .head(30)
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\nPlayers by World Cup team in dataset:")
    print(df["team"].value_counts().head(60).to_string())


if __name__ == "__main__":
    main()