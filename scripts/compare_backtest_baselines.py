import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.dummy import DummyClassifier

from src.data.loaders import DATA_PROCESSED, PROJECT_ROOT
from src.evaluation.metrics import multiclass_brier_score


def main() -> None:
    path = DATA_PROCESSED / "match_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m src.data.build_match_features first."
        )

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    df = df[
        (df["team_a_matches_available_last_10"] > 0)
        & (df["team_b_matches_available_last_10"] > 0)
    ].copy()

    train = df[df["date"] < "2022-01-01"].copy()
    test = df[df["date"] >= "2022-01-01"].copy()

    y_train = train["result"]
    y_test = test["result"]

    rows = []

    dummy_strategies = {
        "most_frequent": DummyClassifier(strategy="most_frequent"),
        "prior": DummyClassifier(strategy="prior"),
        "stratified": DummyClassifier(strategy="stratified", random_state=42),
    }

    # DummyClassifier needs some X, but it ignores it.
    X_train_dummy = train[["neutral"]]
    X_test_dummy = test[["neutral"]]

    for name, model in dummy_strategies.items():
        model.fit(X_train_dummy, y_train)

        pred = model.predict(X_test_dummy)
        proba = model.predict_proba(X_test_dummy)
        classes = model.classes_

        rows.append(
            {
                "model": name,
                "test_matches": len(test),
                "accuracy": accuracy_score(y_test, pred),
                "log_loss": log_loss(y_test, proba, labels=classes),
                "brier_score": multiclass_brier_score(y_test, proba, classes),
            }
        )

    recent_form_metrics_path = (
        PROJECT_ROOT / "outputs" / "tables" / "recent_form_model_backtest_metrics.csv"
    )

    if recent_form_metrics_path.exists():
        recent = pd.read_csv(recent_form_metrics_path).iloc[0].to_dict()
        rows.append(
            {
                "model": "recent_form_logistic_regression",
                "test_matches": int(recent["test_matches"]),
                "accuracy": float(recent["accuracy"]),
                "log_loss": float(recent["log_loss"]),
                "brier_score": float(recent["brier_score"]),
            }
        )

    comparison = pd.DataFrame(rows).sort_values("log_loss")

    output_path = PROJECT_ROOT / "outputs" / "tables" / "backtest_model_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print("\nBacktest model comparison")
    print("-------------------------")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()