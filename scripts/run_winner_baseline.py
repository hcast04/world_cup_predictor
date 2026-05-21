from src.data.loaders import PROJECT_ROOT, load_baseline_data, load_model_elo_ratings
from src.simulation.winner import simulate_world_cup_winner_probabilities


def main() -> None:
    teams, _, fixtures = load_baseline_data()
    elo = load_model_elo_ratings()

    elo_lookup = dict(zip(elo["team"], elo["elo"]))
    host_lookup = dict(zip(teams["team"], teams["is_host"]))

    results, _ = simulate_world_cup_winner_probabilities(
        fixtures=fixtures,
        elo_lookup=elo_lookup,
        host_lookup=host_lookup,
        n_simulations=10_000,
        seed=42,
    )

    output_path = PROJECT_ROOT / "outputs" / "predictions" / "winner_baseline.csv"
    results.to_csv(output_path, index=False)

    print("\nWorld Cup winner baseline")
    print("-------------------------")
    print(results.head(25).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nImportant: this uses an approximate generic knockout bracket.")
    print("Once actual knockout fixtures are known, use manual_knockout_fixtures.csv.")

    print(f"\nSaved results to:")
    print(output_path)


if __name__ == "__main__":
    main()