import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EloPoissonConfig:
    """
    Configuration for a simple Elo-based Poisson goal model.

    base_goals:
        Expected goals per team when both teams are equally strong.

    elo_scale:
        Controls how strongly Elo differences affect expected goals.
        Larger values make Elo differences less influential.

    host_goal_boost:
        Multiplicative boost applied to the host team's expected goals.
        Example: 1.10 means +10% expected goals.
    """

    base_goals: float = 1.35
    elo_scale: float = 600.0
    host_goal_boost: float = 1.10


def expected_goals_from_elo(
    elo_a: float,
    elo_b: float,
    team_a_is_host: bool = False,
    team_b_is_host: bool = False,
    config: EloPoissonConfig | None = None,
) -> tuple[float, float]:
    """
    Convert two Elo ratings into expected goals for team A and team B.

    This is intentionally simple for the first baseline:
        lambda_a = base_goals * exp((elo_a - elo_b) / elo_scale)
        lambda_b = base_goals * exp((elo_b - elo_a) / elo_scale)

    Host teams receive a small expected-goal boost.
    """
    config = config or EloPoissonConfig()

    elo_diff = elo_a - elo_b

    lambda_a = config.base_goals * math.exp(elo_diff / config.elo_scale)
    lambda_b = config.base_goals * math.exp(-elo_diff / config.elo_scale)

    if team_a_is_host:
        lambda_a *= config.host_goal_boost

    if team_b_is_host:
        lambda_b *= config.host_goal_boost

    return lambda_a, lambda_b


def win_draw_loss_probabilities(
    lambda_a: float,
    lambda_b: float,
    max_goals: int = 10,
) -> dict[str, float]:
    """
    Compute approximate win/draw/loss probabilities from two Poisson goal rates.

    Probabilities are computed by summing scoreline probabilities from 0 to max_goals.
    For international football, max_goals=10 is more than enough for a baseline.
    """

    def poisson_pmf(k: int, lam: float) -> float:
        return math.exp(-lam) * (lam**k) / math.factorial(k)

    p_a_win = 0.0
    p_draw = 0.0
    p_b_win = 0.0

    for goals_a in range(max_goals + 1):
        p_goals_a = poisson_pmf(goals_a, lambda_a)

        for goals_b in range(max_goals + 1):
            p_goals_b = poisson_pmf(goals_b, lambda_b)
            p_score = p_goals_a * p_goals_b

            if goals_a > goals_b:
                p_a_win += p_score
            elif goals_a == goals_b:
                p_draw += p_score
            else:
                p_b_win += p_score

    total = p_a_win + p_draw + p_b_win

    return {
        "team_a_win": p_a_win / total,
        "draw": p_draw / total,
        "team_b_win": p_b_win / total,
    }