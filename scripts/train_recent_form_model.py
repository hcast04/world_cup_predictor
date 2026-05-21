import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.data.loaders import DATA_PROCESSED, PROJECT_ROOT
from src.evaluation.metrics import multiclass_brier_score
from src.models.recent_form_model import (
    fit_recent_form_model,
    save_recent_form_model,
)


def main() -> None:
    path = DATA_PROCESSED / "match_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m src.data.build_match_features first."
        )

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    train_df = df[df["date"] < "2026-01-01"].copy()

    model, numeric_features, categorical_features = fit_recent_form_model(train_df)

    # In-sample diagnostics only. Proper evaluation is in backtest_recent_form_model.py.
    filtered = train_df[
        (train_df["team_a_matches_available_last_10"] > 0)
        & (train_df["team_b_matches_available_last_10"] > 0)
    ].copy()

    X = filtered[numeric_features + categorical_features]
    y = filtered["result"]

    proba = model.predict_proba(X)
    pred = model.predict(X)
    classes = model.named_steps["classifier"].classes_

    metrics = {
        "train_matches": len(filtered),
        "accuracy_in_sample": accuracy_score(y, pred),
        "log_loss_in_sample": log_loss(y, proba, labels=classes),
        "brier_score_in_sample": multiclass_brier_score(y, proba, classes),
        "n_numeric_features": len(numeric_features),
        "n_categorical_features": len(categorical_features),
    }

    model_dir = PROJECT_ROOT / "models"
    output_dir = PROJECT_ROOT / "outputs" / "tables"

    model_path = model_dir / "recent_form_model.joblib"
    metadata_path = model_dir / "recent_form_model_features.json"
    metrics_path = output_dir / "recent_form_model_training_metrics.csv"

    save_recent_form_model(
        model=model,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print("\nRecent-form model trained")
    print("-------------------------")
    for key, value in metrics.items():
        if isinstance(value, int):
            print(f"{key}: {value:,}")
        else:
            print(f"{key}: {value:.4f}")

    print("\nSaved:")
    print(model_path)
    print(metadata_path)
    print(metrics_path)


if __name__ == "__main__":
    main()