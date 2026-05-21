from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_PROCESSED
from src.models.team_strength_baseline import fit_team_goal_strengths


def build_team_strengths(
    input_path: Path | None = None,
    output_path: Path | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Build model-ready team goal strengths from historical matches.

    If end_date is provided, only matches before that date are used.
    """
    input_path = input_path or DATA_PROCESSED / "historical_matches.csv"
    output_path = output_path or DATA_PROCESSED / "team_goal_strengths_model.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run python -m src.data.build_historical_matches first."
        )

    matches = pd.read_csv(input_path)
    matches["date"] = pd.to_datetime(matches["date"])

    if end_date is not None:
        matches = matches[matches["date"] < pd.to_datetime(end_date)].copy()

    strengths = fit_team_goal_strengths(matches)
    strengths.to_csv(output_path, index=False)

    return strengths


if __name__ == "__main__":
    strengths = build_team_strengths()
    print(strengths.sort_values("attack_index", ascending=False).head(25).to_string(index=False))
    print(f"\nSaved team strengths to {DATA_PROCESSED / 'team_goal_strengths_model.csv'}")