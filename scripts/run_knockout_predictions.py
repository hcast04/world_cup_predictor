from src.data.load_knockout import load_manual_knockout_fixtures
from src.data.loaders import PROJECT_ROOT, load_baseline_data, load_model_elo_ratings
from src.simulation.knockout import predict_knockout_fixture_probabilities


def main() -> None:
    teams, _, _ = load_baseline_data()
    elo = load_model_elo_ratings()
    fixtures = load_manual_knockout_fixtures()

    if fixtures.empty:
        print("\nNo manual knockout fixtures found.")
        print("Add fixtures to data/raw/manual_knockout_fixtures.csv and rerun.")
        return

    elo_lookup = dict(zip(elo["team"], elo["elo"]))
    host_lookup = dict(zip(teams["team"], teams["is_host"]))

    predictions = predict_knockout_fixture_probabilities(
        fixtures=fixtures,
        elo_lookup=elo_lookup,
        host_lookup=host_lookup,
    )

    output_path = PROJECT_ROOT / "outputs" / "predictions" / "manual_knockout_predictions.csv"
    predictions.to_csv(output_path, index=False)

    print("\nManual knockout predictions")
    print("--------------------------")

    if predictions.empty:
        print("No valid knockout fixtures could be predicted.")
        print("Check for TBD teams or missing Elo ratings.")
    else:
        print(predictions.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nSaved results to:")
    print(output_path)


if __name__ == "__main__":
    main()