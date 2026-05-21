from src.data.load_players import load_players
from src.data.loaders import PROJECT_ROOT, load_baseline_data, load_model_elo_ratings
from src.models.golden_boot import simulate_golden_boot
from src.simulation.winner import simulate_world_cup_winner_probabilities


def main() -> None:
    teams, _, fixtures = load_baseline_data()
    elo = load_model_elo_ratings()
    players = load_players()

    elo_lookup = dict(zip(elo["team"], elo["elo"]))
    host_lookup = dict(zip(teams["team"], teams["is_host"]))

    winner_results, team_matches_samples = simulate_world_cup_winner_probabilities(
        fixtures=fixtures,
        elo_lookup=elo_lookup,
        host_lookup=host_lookup,
        n_simulations=10_000,
        seed=42,
    )

    golden_boot_results = simulate_golden_boot(
        players=players,
        team_matches_samples=team_matches_samples,
        n_simulations=10_000,
        seed=43,
    )

    output_dir = PROJECT_ROOT / "outputs" / "predictions"

    winner_path = output_dir / "stage1_winner_predictions.csv"
    golden_boot_path = output_dir / "stage1_golden_boot_predictions.csv"

    winner_results.to_csv(winner_path, index=False)
    golden_boot_results.to_csv(golden_boot_path, index=False)

    print("\nStage 1: World Cup winner predictions")
    print("-------------------------------------")
    print(winner_results.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nStage 1: Golden Boot predictions")
    print("--------------------------------")
    print(golden_boot_results.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nImportant notes:")
    print("- Group stage is simulated from the current fixture/team/rating inputs.")
    print("- Winner prediction uses an approximate generic knockout bracket for now.")
    print("- Golden Boot uses simulated team match counts from the winner simulation.")
    print("- Replace fake/default ratings and player inputs before treating outputs as meaningful.")

    print(f"\nSaved winner predictions to: {winner_path}")
    print(f"Saved Golden Boot predictions to: {golden_boot_path}")


if __name__ == "__main__":
    main()