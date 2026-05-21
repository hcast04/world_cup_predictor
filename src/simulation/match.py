from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchResult:
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    winner: str | None
    is_draw: bool


def simulate_scoreline(
    team_a: str,
    team_b: str,
    lambda_a: float,
    lambda_b: float,
    rng: np.random.Generator | None = None,
) -> MatchResult:
    """
    Simulate a 90-minute football scoreline using independent Poisson goals.
    """
    rng = rng or np.random.default_rng()

    goals_a = int(rng.poisson(lambda_a))
    goals_b = int(rng.poisson(lambda_b))

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


def simulate_knockout_winner(
    team_a: str,
    team_b: str,
    lambda_a: float,
    lambda_b: float,
    elo_a: float,
    elo_b: float,
    rng: np.random.Generator | None = None,
) -> MatchResult:
    """
    Simulate a knockout match.

    First simulate the 90-minute scoreline. If it is a draw, choose a winner
    using a heavily shrunk Elo-based shootout probability.

    This is a simplification. Later we can add extra time and penalties separately.
    """
    rng = rng or np.random.default_rng()

    result = simulate_scoreline(
        team_a=team_a,
        team_b=team_b,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        rng=rng,
    )

    if not result.is_draw:
        return result

    elo_diff = elo_a - elo_b
    raw_p_a = 1.0 / (1.0 + np.exp(-elo_diff / 400.0))

    # Shrink toward 0.5 because shootouts are much noisier than normal matches.
    p_a_shootout = 0.5 + 0.15 * (raw_p_a - 0.5)

    winner = team_a if rng.random() < p_a_shootout else team_b

    return MatchResult(
        team_a=team_a,
        team_b=team_b,
        goals_a=result.goals_a,
        goals_b=result.goals_b,
        winner=winner,
        is_draw=True,
    )