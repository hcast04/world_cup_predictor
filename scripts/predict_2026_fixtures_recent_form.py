import pandas as pd

from src.data.loaders import DATA_PROCESSED, PROJECT_ROOT
from src.models.recent_form_model import load_recent_form_model


def main() -> None:
    fixture_features_path = DATA_PROCESSED / "fixture_features_2026.csv"
    model_path = PROJECT_ROOT / "models" / "recent_form_model.joblib"
    metadata_path = PROJECT_ROOT / "models" / "recent_form_model_features.json"

    if not fixture_features_path.exists():
        raise FileNotFoundError(
            f"Missing {fixture_features_path}. Run python -m src.data.build_fixture_features first."
        )

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Missing trained recent-form model. Run python scripts/train_recent_form_model.py first."
        )

    fixtures = pd.read_csv(fixture_features_path)
    model, metadata = load_recent_form_model(model_path, metadata_path)

    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]

    required_features = numeric_features + categorical_features
    missing_features = [col for col in required_features if col not in fixtures.columns]

    if missing_features:
        raise ValueError(f"Fixture feature table is missing required columns: {missing_features}")

    X = fixtures[required_features]
    proba = model.predict_proba(X)
    pred = model.predict(X)
    classes = model.named_steps["classifier"].classes_

    out = fixtures[
        [
            "match_id",
            "stage",
            "group",
            "date",
            "team_a",
            "team_b",
            "venue",
            "city",
            "country",
        ]
    ].copy()

    for i, cls in enumerate(classes):
        out[f"prob_{cls}"] = proba[:, i]

    out["predicted_result"] = pred

    # Convenience columns with stable names.
    for cls in ["team_a_win", "draw", "team_b_win"]:
        col = f"prob_{cls}"
        if col not in out.columns:
            out[col] = 0.0

    out["team_a_expected_points"] = (
        3 * out["prob_team_a_win"] + 1 * out["prob_draw"]
    )
    out["team_b_expected_points"] = (
        3 * out["prob_team_b_win"] + 1 * out["prob_draw"]
    )

    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "predictions"
        / "recent_form_fixture_predictions_2026.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print("\nRecent-form fixture predictions")
    print("-------------------------------")
    print(
        out[
            [
                "match_id",
                "group",
                "team_a",
                "team_b",
                "prob_team_a_win",
                "prob_draw",
                "prob_team_b_win",
                "predicted_result",
            ]
        ].head(30).to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print(f"\nSaved predictions to:")
    print(output_path)


if __name__ == "__main__":
    main()