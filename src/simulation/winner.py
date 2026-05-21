import numpy as np
import pandas as pd

from src.models.elo_poisson import expected_goals_from_elo
from src.simulation.match import simulate_knockout_winner
from src.simulation.tournament import simulate_group_stage


def _simulate_knockout_round(
    teams: list[str],
    elo_lookup: dict[str, float],
    host_lookup: dict[str, int],
    rng: np.random.Generator,
) -> list[str]:
    """
    Simulate one generic knockout round.

    Teams are paired in the current list order:
    teams[0] vs teams[1], teams[2] vs teams[3], etc.
    """
    if len(teams) % 2 != 0:
        raise ValueError("Knockout round requires an even number of teams.")

    winners = []

    for i in range(0, len(teams), 2):
        team_a = teams[i]
        team_b = teams[i + 1]

        lambda_a, lambda_b = expected_goals_from_elo(
            elo_a=elo_lookup[team_a],
            elo_b=elo_lookup[team_b],
            team_a_is_host=bool(host_lookup.get(team_a, 0)),
            team_b_is_host=bool(host_lookup.get(team_b, 0)),
        )

        result = simulate_knockout_winner(
            team_a=team_a,
            team_b=team_b,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            elo_a=elo_lookup[team_a],
            elo_b=elo_lookup[team_b],
            rng=rng,
        )

        winners.append(result.winner)

    return winners


def simulate_generic_knockout_winner(
    qualified_teams: list[str],
    elo_lookup: dict[str, float],
    host_lookup: dict[str, int],
    rng: np.random.Generator,
) -> tuple[str, dict[str, int]]:
    """
    Simulate a simplified 32-team knockout bracket.

    This is a placeholder until the official 2026 bracket mapping is encoded.

    Returns:
    - tournament winner
    - team_matches: number of matches played by each team, including group stage
    """
    if len(qualified_teams) != 32:
        raise ValueError(f"Expected 32 qualified teams, got {len(qualified_teams)}")

    # Seed stronger teams apart in a simple way:
    # sort by Elo, then pair high vs low.
    sorted_teams = sorted(
        qualified_teams,
        key=lambda team: elo_lookup.get(team, 1500),
        reverse=True,
    )

    first_round = []
    for i in range(16):
        first_round.append(sorted_teams[i])
        first_round.append(sorted_teams[-(i + 1)])

    team_matches = {team: 3 for team in qualified_teams}

    current_teams = first_round

    while len(current_teams) > 1:
        for team in current_teams:
            team_matches[team] += 1

        winners = _simulate_knockout_round(
            teams=current_teams,
            elo_lookup=elo_lookup,
            host_lookup=host_lookup,
            rng=rng,
        )

        current_teams = winners

    winner = current_teams[0]

    return winner, team_matches


def simulate_world_cup_winner_probabilities(
    fixtures: pd.DataFrame,
    elo_lookup: dict[str, float],
    host_lookup: dict[str, int],
    n_simulations: int = 10_000,
    seed: int = 42,
    manual_results: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    """
    Simulate Stage 1 World Cup winner probabilities.

    This combines:
    - group-stage simulation
    - best-third qualification
    - generic 32-team knockout bracket

    The generic knockout bracket is deliberately marked as approximate.
    """
    rng = np.random.default_rng(seed)

    winner_counts: dict[str, int] = {}
    final_match_counts: dict[str, int] = {}
    semi_final_counts: dict[str, int] = {}
    quarter_final_counts: dict[str, int] = {}
    round_of_16_counts: dict[str, int] = {}
    round_of_32_counts: dict[str, int] = {}

    team_matches_samples = []

    for _ in range(n_simulations):
        _, qualification_table = simulate_group_stage(
            fixtures=fixtures,
            elo_lookup=elo_lookup,
            host_lookup=host_lookup,
            rng=rng,
            skip_incomplete_groups=True,
            manual_results=manual_results,
        )

        qualified = qualification_table.loc[qualification_table["qualifies"], "team"].tolist()

        if len(qualified) != 32:
            # During incomplete-data development, skip invalid tournament samples.
            continue

        winner, team_matches = simulate_generic_knockout_winner(
            qualified_teams=qualified,
            elo_lookup=elo_lookup,
            host_lookup=host_lookup,
            rng=rng,
        )

        team_matches_samples.append(team_matches)

        winner_counts[winner] = winner_counts.get(winner, 0) + 1

        for team, n_matches in team_matches.items():
            if n_matches >= 4:
                round_of_32_counts[team] = round_of_32_counts.get(team, 0) + 1
            if n_matches >= 5:
                round_of_16_counts[team] = round_of_16_counts.get(team, 0) + 1
            if n_matches >= 6:
                quarter_final_counts[team] = quarter_final_counts.get(team, 0) + 1
            if n_matches >= 7:
                semi_final_counts[team] = semi_final_counts.get(team, 0) + 1
            if n_matches >= 8:
                final_match_counts[team] = final_match_counts.get(team, 0) + 1

    valid_sims = len(team_matches_samples)

    if valid_sims == 0:
        raise ValueError(
            "No valid full-tournament simulations were produced. "
            "Check that 32 teams qualify from the simulated group stage."
        )

    all_teams = sorted(set(elo_lookup.keys()))

    rows = []
    for team in all_teams:
        rows.append(
            {
                "team": team,
                "round_of_32_prob": round_of_32_counts.get(team, 0) / valid_sims,
                "round_of_16_prob": round_of_16_counts.get(team, 0) / valid_sims,
                "quarter_final_prob": quarter_final_counts.get(team, 0) / valid_sims,
                "semi_final_prob": semi_final_counts.get(team, 0) / valid_sims,
                "final_prob": final_match_counts.get(team, 0) / valid_sims,
                "winner_prob": winner_counts.get(team, 0) / valid_sims,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        by=["winner_prob", "final_prob", "semi_final_prob"],
        ascending=False,
    )

    return summary, team_matches_samples