import pandas as pd

from src.data.loaders import DATA_RAW, PROJECT_ROOT, load_teams


def main() -> None:
    squads_path = DATA_RAW / "world_cup_squads_2026.csv"

    if not squads_path.exists():
        raise FileNotFoundError(
            f"Missing {squads_path}. Run python -m src.data.collect_world_cup_squads_wikipedia first."
        )

    teams = load_teams()
    squads = pd.read_csv(squads_path)

    if squads.empty:
        print("world_cup_squads_2026.csv is empty.")
        return

    squad_counts = (
        squads.groupby("team")
        .agg(
            squad_players=("player", "count"),
            goalkeepers=("position", lambda s: (s.astype(str).str.upper() == "GK").sum()),
            forwards=("position", lambda s: (s.astype(str).str.upper() == "FW").sum()),
        )
        .reset_index()
    )

    coverage = teams[["team", "group", "confederation"]].merge(
        squad_counts,
        on="team",
        how="left",
    )

    for col in ["squad_players", "goalkeepers", "forwards"]:
        coverage[col] = coverage[col].fillna(0).astype(int)

    coverage["squad_coverage_flag"] = "ok_final_size"

    coverage.loc[
        coverage["squad_players"] == 0,
        "squad_coverage_flag",
    ] = "missing"

    coverage.loc[
        (coverage["squad_players"] > 0) & (coverage["squad_players"] < 23),
        "squad_coverage_flag",
    ] = "incomplete"

    coverage.loc[
        (coverage["squad_players"] > 26),
        "squad_coverage_flag",
    ] = "provisional_or_duplicate"

    coverage.loc[
        (coverage["squad_players"] >= 23) & (coverage["squad_players"] <= 26),
        "squad_coverage_flag",
    ] = "ok_final_size"

    output_path = PROJECT_ROOT / "outputs" / "tables" / "world_cup_squad_coverage.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_path, index=False)

    print("\nWorld Cup squad coverage")
    print("------------------------")
    print(coverage["squad_coverage_flag"].value_counts().to_string())

    print("\nTeams needing attention:")
    needs_work = coverage[coverage["squad_coverage_flag"] != "ok_final_size"]
    if needs_work.empty:
        print("none")
    else:
        print(
            needs_work[
                [
                    "team",
                    "group",
                    "confederation",
                    "squad_players",
                    "goalkeepers",
                    "forwards",
                    "squad_coverage_flag",
                ]
            ].to_string(index=False)
        )

    print(f"\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()