import numpy as np
import pandas as pd

from src.models.match_engine import MatchEngine
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
    match_engine: MatchEngine | None = None,
) -> pd.DataFrame:
    """
    Simulate all unfinished matches for one group and return the ranked table.

    Backward compatible:
    - If match_engine is provided, use it.
    - Otherwise, use the Elo-Poisson model via elo_lookup/host_lookup.
    """
    rng = rng or np.random.default_rng()

    if match_engine is None:
        match_engine = MatchEngine(
            model_type="elo_poisson",
            elo_lookup=elo_lookup,
            host_lookup=host_lookup,
        )

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

        if not match_engine.has_team(team_a) or not match_engine.has_team(team_b):
            continue

        lambda_a, lambda_b = match_engine.expected_goals(team_a, team_b)

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