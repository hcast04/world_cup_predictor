from src.data.loaders import load_baseline_data, load_model_elo_ratings
from src.utils.names import load_team_name_map, normalize_team_column


def main() -> None:
    teams, _, fixtures = load_baseline_data()
    elo = load_model_elo_ratings()
    name_map = load_team_name_map()

    teams = normalize_team_column(teams, "team", name_map)
    elo = normalize_team_column(elo, "team", name_map)
    fixtures = normalize_team_column(fixtures, "team_a", name_map)
    fixtures = normalize_team_column(fixtures, "team_b", name_map)

    print("\nRaw/model data validation")
    print("-------------------------")

    print(f"Teams: {len(teams)}")
    print(f"Elo ratings: {len(elo)}")
    print(f"Fixtures: {len(fixtures)}")

    print("\nBasic structure checks")

    duplicated_teams = teams[teams.duplicated(subset=["team"], keep=False)]
    duplicated_elo = elo[elo.duplicated(subset=["team"], keep=False)]
    duplicated_fixtures = fixtures[fixtures.duplicated(subset=["match_id"], keep=False)]

    print(
        "Duplicate teams:",
        "none" if duplicated_teams.empty else duplicated_teams["team"].tolist(),
    )
    print(
        "Duplicate Elo teams:",
        "none" if duplicated_elo.empty else duplicated_elo["team"].tolist(),
    )
    print(
        "Duplicate fixture IDs:",
        "none" if duplicated_fixtures.empty else duplicated_fixtures["match_id"].tolist(),
    )

    team_names = set(teams["team"])
    elo_names = set(elo["team"])

    missing_elo = sorted(team_names - elo_names)
    elo_without_team = sorted(elo_names - team_names)

    print("\nTeam/Elo checks")
    print(f"Teams missing Elo: {missing_elo if missing_elo else 'none'}")
    print(f"Elo rows without team metadata: {elo_without_team if elo_without_team else 'none'}")

    fixture_teams = set(fixtures["team_a"]).union(set(fixtures["team_b"]))
    fixture_teams = fixture_teams - {"TBD"}

    fixture_teams_missing_metadata = sorted(fixture_teams - team_names)
    fixture_teams_missing_elo = sorted(fixture_teams - elo_names)

    print("\nFixture checks")
    print(
        "Fixture teams missing team metadata:",
        fixture_teams_missing_metadata if fixture_teams_missing_metadata else "none",
    )
    print(
        "Fixture teams missing Elo:",
        fixture_teams_missing_elo if fixture_teams_missing_elo else "none",
    )

    print("\nGroup checks")
    group_counts = teams.groupby("group")["team"].count().sort_index()
    print(group_counts.to_string())

    invalid_group_sizes = group_counts[group_counts != 4]
    print(
        "Groups not containing exactly 4 teams:",
        "none" if invalid_group_sizes.empty else invalid_group_sizes.to_dict(),
    )

    fixture_counts = fixtures[fixtures["stage"] == "group"].groupby("group")["match_id"].count()
    invalid_fixture_counts = fixture_counts[fixture_counts != 6]

    print("\nGroup fixture counts")
    print(fixture_counts.sort_index().to_string())

    print(
        "Groups not containing exactly 6 group fixtures:",
        "none" if invalid_fixture_counts.empty else invalid_fixture_counts.to_dict(),
    )

    has_error = (
        not duplicated_teams.empty
        or not duplicated_elo.empty
        or not duplicated_fixtures.empty
        or bool(missing_elo)
        or bool(fixture_teams_missing_metadata)
        or bool(fixture_teams_missing_elo)
        or not invalid_group_sizes.empty
        or not invalid_fixture_counts.empty
    )

    if has_error:
        raise SystemExit("\nValidation failed. Fix the issues above before running predictions.")

    print("\nValidation passed.")


if __name__ == "__main__":
    main()