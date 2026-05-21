import pandas as pd

from src.data.loaders import DATA_RAW, load_teams, load_elo_ratings


def main() -> None:
    teams = load_teams()
    elo = pd.read_csv(DATA_RAW / "elo_ratings_current.csv")

    if "elo_source" not in elo.columns:
        elo["elo_source"] = "unknown"

    merged = teams.merge(
        elo[["team", "elo", "date", "elo_source"]],
        on="team",
        how="left",
    )

    merged = merged.sort_values(["elo_source", "group", "team"])

    print("\nElo source audit")
    print("----------------")
    print(f"Teams: {len(merged)}")
    print()

    print("Counts by Elo source:")
    print(merged["elo_source"].value_counts(dropna=False).to_string())

    print("\nTeams using temporary confederation defaults:")
    default_teams = merged[merged["elo_source"] == "confederation_default"]

    if default_teams.empty:
        print("none")
    else:
        print(
            default_teams[
                ["team", "group", "confederation", "elo", "elo_source"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()