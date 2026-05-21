import pandas as pd

from src.simulation.match import MatchResult


def match_result_from_row(row: pd.Series) -> MatchResult:
    """
    Convert one manual result row into a MatchResult object.
    """
    team_a = row["team_a"]
    team_b = row["team_b"]
    goals_a = int(row["goals_a"])
    goals_b = int(row["goals_b"])

    if goals_a > goals_b:
        winner = team_a
        is_draw = False
    elif goals_b > goals_a:
        winner = team_b
        is_draw = False
    else:
        winner = None
        is_draw = True

    return MatchResult(
        team_a=team_a,
        team_b=team_b,
        goals_a=goals_a,
        goals_b=goals_b,
        winner=winner,
        is_draw=is_draw,
    )


def build_known_group_results(
    manual_results: pd.DataFrame,
    group_name: str,
) -> list[MatchResult]:
    """
    Build known group-stage MatchResult objects for one group.
    """
    if manual_results.empty:
        return []

    group_results = manual_results[
        (manual_results["stage"] == "group")
        & (manual_results["group"] == group_name)
        & (manual_results["status"] == "played")
    ].copy()

    return [match_result_from_row(row) for _, row in group_results.iterrows()]