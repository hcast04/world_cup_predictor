import pandas as pd

from src.data.loaders import DATA_PROCESSED, PROJECT_ROOT
from src.evaluation.metrics import (
    evaluate_probability_predictions,
    poisson_score_log_loss,
    result_probabilities_from_lambdas,
)
from src.models.team_strength_baseline import (
    fit_team_goal_strengths,
    predict_expected_goals_strength_baseline,
)


def main() -> None:
    matches_path = DATA_PROCESSED / "historical_matches.csv"

    if not matches_path.exists():
        raise FileNotFoundError(
            f"Missing {matches_path}. Run python -m src.data.build_historical_matches first."
        )

    matches = pd.read_csv(matches_path)
    matches["date"] = pd.to_datetime(matches["date"])

    # Simple temporal split:
    # Train on 2010-2021, test on 2022 onwards.
    train = matches[matches["date"] < "2022-01-01"].copy()
    test = matches[matches["date"] >= "2022-01-01"].copy()

    strengths = fit_team_goal_strengths(train)

    rows = []

    for _, row in test.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]

        lambda_a, lambda_b = predict_expected_goals_strength_baseline(
            team_a=team_a,
            team_b=team_b,
            strengths=strengths,
        )

        probs = result_probabilities_from_lambdas(lambda_a, lambda_b)

        rows.append(
            {
                "date": row["date"],
                "team_a": team_a,
                "team_b": team_b,
                "goals_a": row["goals_a"],
                "goals_b": row["goals_b"],
                "result": row["result"],
                "lambda_a": lambda_a,
                "lambda_b": lambda_b,
                **probs,
                "score_log_loss": poisson_score_log_loss(
                    goals_a=int(row["goals_a"]),
                    goals_b=int(row["goals_b"]),
                    lambda_a=lambda_a,
                    lambda_b=lambda_b,
                ),
            }
        )

    predictions = pd.DataFrame(rows)

    metrics = evaluate_probability_predictions(predictions)
    metrics["score_log_loss"] = predictions["score_log_loss"].mean()

    output_dir = PROJECT_ROOT / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "strength_baseline_backtest_predictions.csv"
    metrics_path = output_dir / "strength_baseline_backtest_metrics.csv"
    strengths_path = output_dir / "team_goal_strengths.csv"

    predictions.to_csv(predictions_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    strengths.to_csv(strengths_path, index=False)

    print("\nStrength baseline backtest")
    print("--------------------------")
    print(f"Train matches: {len(train):,}")
    print(f"Test matches: {len(test):,}")
    print()

    for key, value in metrics.items():
        if key == "n_matches":
            print(f"{key}: {value:,}")
        else:
            print(f"{key}: {value:.4f}")

    print("\nSaved:")
    print(predictions_path)
    print(metrics_path)
    print(strengths_path)


if __name__ == "__main__":
    main()