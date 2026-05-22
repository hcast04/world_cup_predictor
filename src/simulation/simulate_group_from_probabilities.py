import numpy as np
import pandas as pd

from src.simulation.group_stage import build_group_table
from src.simulation.probability_match import simulate_match_from_result_probabilities


def simulate_group_from_fixture_probabilities(
    group_name: str,
    fixture_predictions: pd.DataFrame,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Simulate one group from fixture-level W/D/L probabilities.

    Expected columns:
    - group
    - team_a
    - team_b
    - prob_team_a_win
    - prob_draw
    - prob_team_b_win
    """
    rng = rng or np.random.default_rng()

    group_predictions = fixture_predictions[
        fixture_predictions["group"] == group_name
    ].copy()

    if group_predictions.empty:
        raise ValueError(f"No fixture predictions found for group {group_name}")

    teams = sorted(
        set(group_predictions["team_a"]).union(set(group_predictions["team_b"]))
    )

    results = []

    for _, row in group_predictions.iterrows():
        result = simulate_match_from_result_probabilities(
            team_a=row["team_a"],
            team_b=row["team_b"],
            p_team_a_win=float(row["prob_team_a_win"]),
            p_draw=float(row["prob_draw"]),
            p_team_b_win=float(row["prob_team_b_win"]),
            rng=rng,
        )

        results.append(result)

    return build_group_table(teams=teams, results=results)


def simulate_group_stage_from_fixture_probabilities(
    fixture_predictions: pd.DataFrame,
    rng: np.random.Generator | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Simulate all groups from fixture-level W/D/L probabilities.
    """
    rng = rng or np.random.default_rng()

    groups = sorted(fixture_predictions["group"].dropna().unique())

    group_tables = {}

    for group_name in groups:
        group_tables[group_name] = simulate_group_from_fixture_probabilities(
            group_name=group_name,
            fixture_predictions=fixture_predictions,
            rng=rng,
        )

    return group_tables