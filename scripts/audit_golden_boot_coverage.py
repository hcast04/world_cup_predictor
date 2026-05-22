import pandas as pd

from src.data.loaders import DATA_PROCESSED, PROJECT_ROOT, load_teams


def main() -> None:
    teams = load_teams()

    scoring_path = DATA_PROCESSED / "player_scoring_features.csv"
    pool_path = DATA_PROCESSED / "golden_boot_player_pool.csv"

    if not scoring_path.exists():
        raise FileNotFoundError(
            f"Missing {scoring_path}. Run python -m src.data.build_player_scoring_features first."
        )

    if not pool_path.exists():
        raise FileNotFoundError(
            f"Missing {pool_path}. Run python -m src.data.build_golden_boot_pool first."
        )

    scoring = pd.read_csv(scoring_path)
    pool = pd.read_csv(pool_path)

    scoring_counts = (
        scoring[scoring["team"].isin(set(teams["team"]))]
        .groupby("team")
        .agg(
            club_dataset_players=("player", "count"),
            club_dataset_attackers=(
                "position",
                lambda s: s.astype(str).str.contains("FW").sum(),
            ),
            max_minutes=("minutes", "max"),
            max_xg=("xg", "max"),
            max_goals=("goals", "max"),
        )
        .reset_index()
    )

    pool_counts = (
        pool.groupby("team")
        .agg(
            golden_boot_candidates=("player", "count"),
            top_candidate_score=("club_scoring_score_shrunk", "max"),
            top_candidate_xg=("xg", "max"),
            top_candidate_goals=("goals", "max"),
        )
        .reset_index()
    )

    coverage = teams[["team", "group", "confederation"]].merge(
        scoring_counts,
        on="team",
        how="left",
    )

    coverage = coverage.merge(
        pool_counts,
        on="team",
        how="left",
    )

    numeric_cols = [
        "club_dataset_players",
        "club_dataset_attackers",
        "max_minutes",
        "max_xg",
        "max_goals",
        "golden_boot_candidates",
        "top_candidate_score",
        "top_candidate_xg",
        "top_candidate_goals",
    ]

    for col in numeric_cols:
        coverage[col] = coverage[col].fillna(0)

    coverage["coverage_flag"] = "ok"
    coverage.loc[coverage["club_dataset_players"] == 0, "coverage_flag"] = "no_club_data"
    coverage.loc[
        (coverage["club_dataset_players"] > 0)
        & (coverage["golden_boot_candidates"] == 0),
        "coverage_flag",
    ] = "no_golden_boot_candidates"
    coverage.loc[
        (coverage["golden_boot_candidates"] > 0)
        & (coverage["golden_boot_candidates"] < 3),
        "coverage_flag",
    ] = "low_candidate_count"

    coverage = coverage.sort_values(
        by=["coverage_flag", "group", "team"],
        ascending=[True, True, True],
    )

    output_path = PROJECT_ROOT / "outputs" / "tables" / "golden_boot_coverage_audit.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_path, index=False)

    print("\nGolden Boot data coverage audit")
    print("-------------------------------")
    print("Coverage flags:")
    print(coverage["coverage_flag"].value_counts().to_string())

    print("\nTeams with weakest Golden Boot data coverage:")
    weak = coverage[coverage["coverage_flag"] != "ok"].copy()

    if weak.empty:
        print("none")
    else:
        print(
            weak[
                [
                    "team",
                    "group",
                    "confederation",
                    "club_dataset_players",
                    "club_dataset_attackers",
                    "golden_boot_candidates",
                    "coverage_flag",
                ]
            ].to_string(index=False)
        )

    print(f"\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()