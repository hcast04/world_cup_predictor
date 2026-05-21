import math

import pandas as pd

from src.models.elo_poisson import win_draw_loss_probabilities


def brier_score_three_way(
    actual_result: str,
    p_team_a_win: float,
    p_draw: float,
    p_team_b_win: float,
) -> float:
    """
    Three-class Brier score for team_a_win/draw/team_b_win.
    Lower is better.
    """
    actual = {
        "team_a_win": 0.0,
        "draw": 0.0,
        "team_b_win": 0.0,
    }
    actual[actual_result] = 1.0

    return (
        (p_team_a_win - actual["team_a_win"]) ** 2
        + (p_draw - actual["draw"]) ** 2
        + (p_team_b_win - actual["team_b_win"]) ** 2
    )


def log_loss_three_way(
    actual_result: str,
    p_team_a_win: float,
    p_draw: float,
    p_team_b_win: float,
    eps: float = 1e-12,
) -> float:
    """
    Three-class log loss for team_a_win/draw/team_b_win.
    Lower is better.
    """
    probs = {
        "team_a_win": p_team_a_win,
        "draw": p_draw,
        "team_b_win": p_team_b_win,
    }

    p = max(min(probs[actual_result], 1.0 - eps), eps)
    return -math.log(p)


def evaluate_probability_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    """
    Evaluate match-result probability predictions.

    Expected columns:
    - result
    - team_a_win_prob
    - draw_prob
    - team_b_win_prob
    """
    brier_scores = []
    log_losses = []
    correct = []

    for _, row in predictions.iterrows():
        result = row["result"]

        p_a = float(row["team_a_win_prob"])
        p_d = float(row["draw_prob"])
        p_b = float(row["team_b_win_prob"])

        brier_scores.append(brier_score_three_way(result, p_a, p_d, p_b))
        log_losses.append(log_loss_three_way(result, p_a, p_d, p_b))

        predicted_result = max(
            {
                "team_a_win": p_a,
                "draw": p_d,
                "team_b_win": p_b,
            },
            key={
                "team_a_win": p_a,
                "draw": p_d,
                "team_b_win": p_b,
            }.get,
        )
        correct.append(predicted_result == result)

    return {
        "n_matches": len(predictions),
        "brier_score": sum(brier_scores) / len(brier_scores),
        "log_loss": sum(log_losses) / len(log_losses),
        "accuracy": sum(correct) / len(correct),
    }


def poisson_score_log_loss(
    goals_a: int,
    goals_b: int,
    lambda_a: float,
    lambda_b: float,
    eps: float = 1e-12,
) -> float:
    """
    Negative log likelihood of the observed score under independent Poisson goals.
    """
    def poisson_pmf(k: int, lam: float) -> float:
        return math.exp(-lam) * (lam**k) / math.factorial(k)

    p_score = poisson_pmf(goals_a, lambda_a) * poisson_pmf(goals_b, lambda_b)
    return -math.log(max(p_score, eps))


def result_probabilities_from_lambdas(lambda_a: float, lambda_b: float) -> dict[str, float]:
    probs = win_draw_loss_probabilities(lambda_a, lambda_b)

    return {
        "team_a_win_prob": probs["team_a_win"],
        "draw_prob": probs["draw"],
        "team_b_win_prob": probs["team_b_win"],
    }