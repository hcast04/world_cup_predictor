from src.data.load_players import load_players
from src.data.loaders import PROJECT_ROOT, load_teams
from src.models.golden_boot import simulate_golden_boot


def main() -> None:
    players = load_players()
    teams = load_teams()

    # Temporary assumption:
    # every team plays exactly 3 matches.
    # Later this will come from group-stage and knockout simulations.
    team_matches = {team: 3 for team in teams["team"]}

    results = simulate_golden_boot(
        players=players,
        team_matches_samples=[team_matches],
        n_simulations=10_000,
        seed=42,
    )

    output_path = PROJECT_ROOT / "outputs" / "predictions" / "golden_boot_baseline.csv"
    results.to_csv(output_path, index=False)

    print("\nGolden Boot baseline")
    print("--------------------")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nSaved results to:")
    print(output_path)


if __name__ == "__main__":
    main()