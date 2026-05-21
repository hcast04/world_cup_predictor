from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_PROCESSED


def _points_for_result(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _build_team_match_history(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level team_a/team_b rows into one row per team per match.
    """
    team_a = matches.rename(
        columns={
            "team_a": "team",
            "team_b": "opponent",
            "goals_a": "goals_for",
            "goals_b": "goals_against",
        }
    ).copy()
    team_a["is_team_a"] = 1

    team_b = matches.rename(
        columns={
            "team_b": "team",
            "team_a": "opponent",
            "goals_b": "goals_for",
            "goals_a": "goals_against",
        }
    ).copy()
    team_b["is_team_a"] = 0

    keep_columns = [
        "date",
        "team",
        "opponent",
        "goals_for",
        "goals_against",
        "tournament",
        "neutral",
        "is_team_a",
    ]

    long = pd.concat(
        [team_a[keep_columns], team_b[keep_columns]],
        ignore_index=True,
    )

    long["points"] = [
        _points_for_result(gf, ga)
        for gf, ga in zip(long["goals_for"], long["goals_against"])
    ]
    long["win"] = (long["goals_for"] > long["goals_against"]).astype(int)
    long["draw"] = (long["goals_for"] == long["goals_against"]).astype(int)
    long["loss"] = (long["goals_for"] < long["goals_against"]).astype(int)
    long["goal_diff"] = long["goals_for"] - long["goals_against"]

    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    return long


def _add_rolling_features(
    team_history: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """
    Add rolling recent-form features for each team.

    Important: we shift by one match so the current match is not used to predict itself.
    """
    df = team_history.copy()
    grouped = df.groupby("team", group_keys=False)

    base_columns = [
        "goals_for",
        "goals_against",
        "points",
        "win",
        "draw",
        "loss",
        "goal_diff",
    ]

    for window in windows:
        for col in base_columns:
            feature_name = f"{col}_last_{window}"

            df[feature_name] = grouped[col].transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
            )

        df[f"matches_available_last_{window}"] = grouped["points"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).count()
        )

    return df


def build_match_features(
    input_path: Path | None = None,
    output_path: Path | None = None,
    windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """
    Build match-level features from historical matches.

    Output: one row per match, with recent-form features for team_a and team_b.
    """
    input_path = input_path or DATA_PROCESSED / "historical_matches.csv"
    output_path = output_path or DATA_PROCESSED / "match_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run python -m src.data.build_historical_matches first."
        )

    matches = pd.read_csv(input_path)
    matches["date"] = pd.to_datetime(matches["date"])

    matches = matches.sort_values("date").reset_index(drop=True)
    matches["match_key"] = matches.index

    team_history = _build_team_match_history(matches)
    team_history = team_history.merge(
        matches[["match_key", "date", "team_a", "team_b"]],
        left_on=["date", "team"],
        right_on=["date", "team_a"],
        how="left",
    )

    # The merge above only works for team_a rows. Build keys more robustly instead.
    team_a_history = team_history[team_history["is_team_a"] == 1].copy()
    team_b_history = team_history[team_history["is_team_a"] == 0].copy()

    # Reconstruct match_key by date/team/opponent matching.
    team_a_keys = matches[["match_key", "date", "team_a", "team_b"]].rename(
        columns={"team_a": "team", "team_b": "opponent"}
    )
    team_b_keys = matches[["match_key", "date", "team_a", "team_b"]].rename(
        columns={"team_b": "team", "team_a": "opponent"}
    )

    team_keys = pd.concat([team_a_keys, team_b_keys], ignore_index=True)

    team_history = _build_team_match_history(matches)
    team_history = team_history.merge(
        team_keys,
        on=["date", "team", "opponent"],
        how="left",
    )

    if team_history["match_key"].isna().any():
        missing = team_history[team_history["match_key"].isna()].head(10)
        raise ValueError(
            "Could not assign match_key to some team-history rows:\n"
            f"{missing.to_string(index=False)}"
        )

    team_history["match_key"] = team_history["match_key"].astype(int)
    team_history = _add_rolling_features(team_history, windows=windows)

    feature_cols = []
    for window in windows:
        feature_cols.extend(
            [
                f"goals_for_last_{window}",
                f"goals_against_last_{window}",
                f"points_last_{window}",
                f"win_last_{window}",
                f"draw_last_{window}",
                f"loss_last_{window}",
                f"goal_diff_last_{window}",
                f"matches_available_last_{window}",
            ]
        )

    team_a_features = team_history[team_history["is_team_a"] == 1][
        ["match_key"] + feature_cols
    ].copy()
    team_a_features = team_a_features.rename(
        columns={col: f"team_a_{col}" for col in feature_cols}
    )

    team_b_features = team_history[team_history["is_team_a"] == 0][
        ["match_key"] + feature_cols
    ].copy()
    team_b_features = team_b_features.rename(
        columns={col: f"team_b_{col}" for col in feature_cols}
    )

    out = matches.merge(team_a_features, on="match_key", how="left")
    out = out.merge(team_b_features, on="match_key", how="left")

    for window in windows:
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

    out = out.sort_values("date").reset_index(drop=True)
    out.to_csv(output_path, index=False)

    return out


if __name__ == "__main__":
    features = build_match_features()
    print(features.head().to_string(index=False))
    print()
    print(features.tail().to_string(index=False))
    print(f"\nSaved {len(features):,} rows to {DATA_PROCESSED / 'match_features.csv'}")