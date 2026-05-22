import numpy as np

from src.simulation.match import MatchResult


DRAW_SCORELINES = [
    (0, 0, 0.35),
    (1, 1, 0.50),
    (2, 2, 0.13),
    (3, 3, 0.02),
]

WIN_SCORELINES = [
    (1, 0, 0.35),
    (2, 0, 0.20),
    (2, 1, 0.25),
    (3, 0, 0.07),
    (3, 1, 0.09),
    (3, 2, 0.03),
    (4, 1, 0.01),
]


def _sample_weighted_scoreline(
    scorelines: list[tuple[int, int, float]],
    rng: np.random.Generator,
) -> tuple[int, int]:
    scores = [(a, b) for a, b, _ in scorelines]
    weights = np.array([w for _, _, w in scorelines], dtype=float)
    weights = weights / weights.sum()

    idx = int(rng.choice(len(scores), p=weights))

    return scores[idx]


def simulate_match_from_result_probabilities(
    team_a: str,
    team_b: str,
    p_team_a_win: float,
    p_draw: float,
    p_team_b_win: float,
    rng: np.random.Generator | None = None,
) -> MatchResult:
    """
    Simulate a match result from W/D/L probabilities.

    Since the recent-form model only predicts outcome probabilities, this function
    samples a plausible scoreline conditional on the sampled result.
    """
    rng = rng or np.random.default_rng()

    probs = np.array([p_team_a_win, p_draw, p_team_b_win], dtype=float)
    probs = probs / probs.sum()

    outcome = rng.choice(["team_a_win", "draw", "team_b_win"], p=probs)

    if outcome == "team_a_win":
        goals_a, goals_b = _sample_weighted_scoreline(WIN_SCORELINES, rng)

        return MatchResult(
            team_a=team_a,
            team_b=team_b,
            goals_a=goals_a,
            goals_b=goals_b,
            winner=team_a,
            is_draw=False,
        )

    if outcome == "team_b_win":
        goals_b, goals_a = _sample_weighted_scoreline(WIN_SCORELINES, rng)

        return MatchResult(
            team_a=team_a,
            team_b=team_b,
            goals_a=goals_a,
            goals_b=goals_b,
            winner=team_b,
            is_draw=False,
        )

    goals_a, goals_b = _sample_weighted_scoreline(DRAW_SCORELINES, rng)

    return MatchResult(
        team_a=team_a,
        team_b=team_b,
        goals_a=goals_a,
        goals_b=goals_b,
        winner=None,
        is_draw=True,
    )