import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "tournament",
]


def get_available_features(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Return the available numeric and categorical features.
    """
    numeric = [col for col in NUMERIC_FEATURES if col in df.columns]
    categorical = [col for col in CATEGORICAL_FEATURES if col in df.columns]

    return numeric, categorical


def build_recent_form_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Build the recent-form multinomial logistic-regression pipeline.
    """
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

    return model


def fit_recent_form_model(
    match_features: pd.DataFrame,
) -> tuple[Pipeline, list[str], list[str]]:
    """
    Fit the recent-form model on a match feature table.
    """
    df = match_features.copy()

    df = df[
        (df["team_a_matches_available_last_10"] > 0)
        & (df["team_b_matches_available_last_10"] > 0)
    ].copy()

    numeric_features, categorical_features = get_available_features(df)

    X = df[numeric_features + categorical_features]
    y = df["result"]

    model = build_recent_form_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    model.fit(X, y)

    return model, numeric_features, categorical_features


def save_recent_form_model(
    model: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
    model_path: Path,
    metadata_path: Path,
) -> None:
    """
    Save fitted model and feature metadata.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    metadata = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": "result",
        "classes": list(model.named_steps["classifier"].classes_),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_recent_form_model(
    model_path: Path,
    metadata_path: Path,
) -> tuple[Pipeline, dict]:
    """
    Load fitted model and metadata.
    """
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return model, metadata