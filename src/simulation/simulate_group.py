import numpy as np
import pandas as pd

from src.models.elo_poisson import expected_goals_from_elo
from src.simulation.group_stage import build_group_table
from src.simulation.match import simulate_scoreline


def simulate_group_from_fixtures(
    group_name: str,
    fixtures: pd.DataFrame,
    elo_lookup: dict[str, float],
    host_lookup: dict[str, int],
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Simulate all matches for one group and return the ranked group table.

    This assumes fixtures contains only group-stage rows.
    Rows with TBD teams are skipped.
    """
    rng = rng or np.random.default_rng()

    group_fixtures = fixtures[
        (fixtures["stage"] == "group") & (fixtures["group"] == group_name)
    ].copy()

    if group_fixtures.empty:
        raise ValueError(f"No fixtures found for group {group_name}")

    teams = sorted(
        set(group_fixtures["team_a"]).union(set(group_fixtures["team_b"])) - {"TBD"}
    )

    if len(teams) == 0:
        raise ValueError(f"No known teams found for group {group_name}")

    results = []

    for _, row in group_fixtures.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]

        if team_a == "TBD" or team_b == "TBD":
            continue

        if team_a not in elo_lookup or team_b not in elo_lookup:
            continue

        lambda_a, lambda_b = expected_goals_from_elo(
            elo_a=elo_lookup[team_a],
            elo_b=elo_lookup[team_b],
            team_a_is_host=bool(host_lookup.get(team_a, 0)),
            team_b_is_host=bool(host_lookup.get(team_b, 0)),
        )

        result = simulate_scoreline(
            team_a=team_a,
            team_b=team_b,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            rng=rng,
        )

        results.append(result)

    if len(results) == 0:
        raise ValueError(f"No simulatable matches found for group {group_name}")

    return build_group_table(teams=teams, results=results)