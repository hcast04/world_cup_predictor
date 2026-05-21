import argparse

from src.data.load_knockout import load_manual_knockout_fixtures
from src.data.load_players import load_players
from src.data.loaders import PROJECT_ROOT, load_baseline_data, load_model_elo_ratings
from src.models.golden_boot import simulate_golden_boot
from src.simulation.knockout import predict_knockout_fixture_probabilities
from src.simulation.winner import simulate_world_cup_winner_probabilities


def run_stage_1(n_simulations: int, seed: int) -> None:
    """
    Stage 1:
    - Predict World Cup winner
    - Predict Golden Boot
    - Uses group-stage simulation and approximate generic knockout bracket
    """
    teams, _, fixtures = load_baseline_data()
    elo = load_model_elo_ratings()
    players = load_players()

    elo_lookup = dict(zip(elo["team"], elo["elo"]))
    host_lookup = dict(zip(teams["team"], teams["is_host"]))

    winner_results, team_matches_samples = simulate_world_cup_winner_probabilities(
        fixtures=fixtures,
        elo_lookup=elo_lookup,
        host_lookup=host_lookup,
        n_simulations=n_simulations,
        seed=seed,
    )

    golden_boot_results = simulate_golden_boot(
        players=players,
        team_matches_samples=team_matches_samples,
        n_simulations=n_simulations,
        seed=seed + 1,
    )

    output_dir = PROJECT_ROOT / "outputs" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    winner_path = output_dir / "stage1_winner_predictions.csv"
    golden_boot_path = output_dir / "stage1_golden_boot_predictions.csv"

    winner_results.to_csv(winner_path, index=False)
    golden_boot_results.to_csv(golden_boot_path, index=False)

    print("\nStage 1 complete")
    print("----------------")
    print("World Cup winner predictions:")
    print(winner_results.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nGolden Boot predictions:")
    print(golden_boot_results.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nSaved:")
    print(winner_path)
    print(golden_boot_path)

    print("\nNotes:")
    print("- Winner prediction currently uses an approximate generic knockout bracket.")
    print("- Golden Boot uses simulated team match counts from the winner simulation.")
    print("- Replace default Elo/player seed data before treating outputs as meaningful.")


def run_stage_2_or_3(stage: int) -> None:
    """
    Stage 2 / Stage 3:
    - Read manually entered knockout fixtures
    - Predict each fixture independently
    """
    teams, _, _ = load_baseline_data()
    elo = load_model_elo_ratings()
    fixtures = load_manual_knockout_fixtures()

    if fixtures.empty:
        print(f"\nStage {stage}: no manual knockout fixtures found.")
        print("Add fixtures to data/raw/manual_knockout_fixtures.csv and rerun.")
        return

    if stage == 2:
        relevant_stages = {"round_of_32"}
    else:
        relevant_stages = {
            "round_of_16",
            "quarter_final",
            "semi_final",
            "third_place",
            "final",
        }

    fixtures = fixtures[fixtures["stage"].isin(relevant_stages)].copy()

    if fixtures.empty:
        print(f"\nStage {stage}: no fixtures found for expected stages.")
        print(f"Expected one of: {sorted(relevant_stages)}")
        return

    elo_lookup = dict(zip(elo["team"], elo["elo"]))
    host_lookup = dict(zip(teams["team"], teams["is_host"]))

    predictions = predict_knockout_fixture_probabilities(
        fixtures=fixtures,
        elo_lookup=elo_lookup,
        host_lookup=host_lookup,
    )

    output_dir = PROJECT_ROOT / "outputs" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"stage{stage}_knockout_predictions.csv"
    predictions.to_csv(output_path, index=False)

    print(f"\nStage {stage} knockout predictions")
    print("----------------------------------")

    if predictions.empty:
        print("No valid fixtures could be predicted.")
        print("Check for TBD teams or missing Elo ratings.")
    else:
        print(predictions.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nSaved:")
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run World Cup prediction stages.")
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Prediction stage to run: 1, 2, or 3.",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=10_000,
        help="Number of Monte Carlo simulations for Stage 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    if args.stage == 1:
        run_stage_1(n_simulations=args.n_simulations, seed=args.seed)
    elif args.stage in {2, 3}:
        run_stage_2_or_3(stage=args.stage)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()