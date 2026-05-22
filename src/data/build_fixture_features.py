from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_PROCESSED, load_fixtures
from src.utils.names import load_team_name_map, normalize_team_column


FORM_FEATURE_COLUMNS = [
    "goals_for_last_5",
    "goals_against_last_5",
    "points_last_5",
    "win_last_5",
    "draw_last_5",
    "loss_last_5",
    "goal_diff_last_5",
    "matches_available_last_5",
    "goals_for_last_10",
    "goals_against_last_10",
    "points_last_10",
    "win_last_10",
    "draw_last_10",
    "loss_last_10",
    "goal_diff_last_10",
    "matches_available_last_10",
]


def _latest_team_form(match_features: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the latest available rolling-form row for each team.

    match_features is match-level. We reconstruct team-level form from both
    team_a_* and team_b_* columns, then keep each team's latest row.
    """
    rows = []

    for side in ["team_a", "team_b"]:
        team_col = side

        side_cols = {
            f"{side}_{feature}": feature
            for feature in FORM_FEATURE_COLUMNS
            if f"{side}_{feature}" in match_features.columns
        }

        if not side_cols:
            raise ValueError(f"No rolling-form columns found for {side}")

        tmp = match_features[["date", team_col] + list(side_cols.keys())].copy()
        tmp = tmp.rename(columns={team_col: "team", **side_cols})
        rows.append(tmp)

    long = pd.concat(rows, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])

    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    latest = long.groupby("team", as_index=False).tail(1).reset_index(drop=True)

    return latest


def build_fixture_features(
    fixtures_path: Path | None = None,
    match_features_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build recent-form features for future 2026 fixtures.

    Uses the latest available rolling-form features for each team.
    """
    match_features_path = match_features_path or DATA_PROCESSED / "match_features.csv"
    output_path = output_path or DATA_PROCESSED / "fixture_features_2026.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not match_features_path.exists():
        raise FileNotFoundError(
            f"Missing {match_features_path}. Run python -m src.data.build_match_features first."
        )

    fixtures = load_fixtures(fixtures_path)
    fixtures = fixtures[fixtures["stage"] == "group"].copy()

    name_map = load_team_name_map()
    fixtures = normalize_team_column(fixtures, "team_a", name_map)
    fixtures = normalize_team_column(fixtures, "team_b", name_map)

    fixtures = fixtures[
        (fixtures["team_a"] != "TBD")
        & (fixtures["team_b"] != "TBD")
    ].copy()

    match_features = pd.read_csv(match_features_path)
    match_features["date"] = pd.to_datetime(match_features["date"])

    latest_form = _latest_team_form(match_features)

    team_a_form = latest_form.rename(
        columns={
            "team": "team_a",
            **{col: f"team_a_{col}" for col in FORM_FEATURE_COLUMNS if col in latest_form.columns},
        }
    )

    team_b_form = latest_form.rename(
        columns={
            "team": "team_b",
            **{col: f"team_b_{col}" for col in FORM_FEATURE_COLUMNS if col in latest_form.columns},
        }
    )

    out = fixtures.merge(team_a_form, on="team_a", how="left")
    out = out.merge(team_b_form, on="team_b", how="left")

    missing_team_a = out[out["team_a_points_last_10"].isna()]["team_a"].unique().tolist()
    missing_team_b = out[out["team_b_points_last_10"].isna()]["team_b"].unique().tolist()

    missing_teams = sorted(set(missing_team_a + missing_team_b))

    if missing_teams:
        print("\nWarning: missing recent-form data for teams:")
        for team in missing_teams:
            print(f"- {team}")

    for window in [5, 10]:
        out[f"points_diff_last_{window}"] = (
            out[f"team_a_points_last_{window}"] - out[f"team_b_points_last_{window}"]
        )
        out[f"goal_diff_diff_last_{window}"] = (
            out[f"team_a_goal_diff_last_{window}"] - out[f"team_b_goal_diff_last_{window}"]
        )
        out[f"goals_for_diff_last_{window}"] = (
            out[f"team_a_goals_for_last_{window}"] - out[f"team_b_goals_for_last_{window}"]
        )
        out[f"goals_against_diff_last_{window}"] = (
            out[f"team_a_goals_against_last_{window}"]
            - out[f"team_b_goals_against_last_{window}"]
        )
        out[f"win_rate_diff_last_{window}"] = (
            out[f"team_a_win_last_{window}"] - out[f"team_b_win_last_{window}"]
        )

    # The trained recent-form model expects these columns.
    if "neutral" not in out.columns:
        out["neutral"] = True

    out["tournament"] = "FIFA World Cup"

    out = out.sort_values(["group", "match_id"]).reset_index(drop=True)
    out.to_csv(output_path, index=False)

    return out


if __name__ == "__main__":
    features = build_fixture_features()
    print(features.head(20).to_string(index=False))
    print(f"\nSaved {len(features):,} fixture rows to {DATA_PROCESSED / 'fixture_features_2026.csv'}")