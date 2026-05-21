from src.data.loaders import load_baseline_data
from src.utils.names import load_team_name_map, normalize_team_column


def main() -> None:
    teams, elo, fixtures = load_baseline_data()
    name_map = load_team_name_map()

    teams = normalize_team_column(teams, "team", name_map)
    elo = normalize_team_column(elo, "team", name_map)
    fixtures = normalize_team_column(fixtures, "team_a", name_map)
    fixtures = normalize_team_column(fixtures, "team_b", name_map)

    print("\nRaw data validation")
    print("-------------------")

    print(f"Teams: {len(teams)}")
    print(f"Elo ratings: {len(elo)}")
    print(f"Fixtures: {len(fixtures)}")

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


if __name__ == "__main__":
    main()