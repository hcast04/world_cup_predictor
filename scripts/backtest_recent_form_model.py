import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.evaluation.metrics import multiclass_brier_score
from src.data.loaders import DATA_PROCESSED, PROJECT_ROOT




def main() -> None:
    path = DATA_PROCESSED / "match_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m src.data.build_match_features first."
        )

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    # Drop very early rows where teams have no prior match history at all.
    df = df[
        (df["team_a_matches_available_last_10"] > 0)
        & (df["team_b_matches_available_last_10"] > 0)
    ].copy()

    train = df[df["date"] < "2022-01-01"].copy()
    test = df[df["date"] >= "2022-01-01"].copy()

    numeric_features = [
        "points_diff_last_5",
        "goal_diff_diff_last_5",
        "goals_for_diff_last_5",
        "goals_against_diff_last_5",
        "win_rate_diff_last_5",
        "points_diff_last_10",
        "goal_diff_diff_last_10",
        "goals_for_diff_last_10",
        "goals_against_diff_last_10",
        "win_rate_diff_last_10",
        "neutral",
    ]

    categorical_features = ["tournament"]

    numeric_features = [col for col in numeric_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]

    X_train = train[numeric_features + categorical_features]
    y_train = train["result"]

    X_test = test[numeric_features + categorical_features]
    y_test = test["result"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)
    pred = model.predict(X_test)
    classes = model.named_steps["classifier"].classes_

    metrics = {
        "train_matches": len(train),
        "test_matches": len(test),
        "accuracy": accuracy_score(y_test, pred),
        "log_loss": log_loss(y_test, proba, labels=classes),
        "brier_score": multiclass_brier_score(y_test, proba, classes),
    }

    pred_df = test[
        [
            "date",
            "team_a",
            "team_b",
            "goals_a",
            "goals_b",
            "result",
            "tournament",
        ]
    ].copy()

    for i, cls in enumerate(classes):
        pred_df[f"prob_{cls}"] = proba[:, i]

    pred_df["predicted_result"] = pred

    output_dir = PROJECT_ROOT / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / "recent_form_model_backtest_predictions.csv"
    metrics_path = output_dir / "recent_form_model_backtest_metrics.csv"

    pred_df.to_csv(pred_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print("\nRecent-form model backtest")
    print("--------------------------")
    for key, value in metrics.items():
        if isinstance(value, int):
            print(f"{key}: {value:,}")
        else:
            print(f"{key}: {value:.4f}")

    print("\nClass order:")
    print(classes)

    print("\nSaved:")
    print(pred_path)
    print(metrics_path)


if __name__ == "__main__":
    main()