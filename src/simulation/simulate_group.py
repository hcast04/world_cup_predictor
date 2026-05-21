import numpy as np
import pandas as pd

from src.models.elo_poisson import expected_goals_from_elo
from src.simulation.group_stage import build_group_table
from src.simulation.known_results import build_known_group_results
from src.simulation.match import simulate_scoreline


def simulate_group_from_fixtures(
    group_name: str,
    fixtures: pd.DataFrame,
    elo_lookup: dict[str, float],
    host_lookup: dict[str, int],
    rng: np.random.Generator | None = None,
    manual_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Simulate all unfinished matches for one group and return the ranked table.

    If manual_results contains already-played matches for this group, those
    results are used directly instead of being simulated.

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

    known_results = []
    known_match_ids = set()

    if manual_results is not None and not manual_results.empty:
        known_group_results = manual_results[
            (manual_results["stage"] == "group")
            & (manual_results["group"] == group_name)
            & (manual_results["status"] == "played")
        ].copy()

        known_match_ids = set(known_group_results["match_id"].astype(str))
        known_results = build_known_group_results(manual_results, group_name)

    results.extend(known_results)

    for _, row in group_fixtures.iterrows():
        match_id = str(row["match_id"])

        if match_id in known_match_ids:
            continue

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
        raise ValueError(f"No simulatable or known matches found for group {group_name}")

    return build_group_table(teams=teams, results=results)