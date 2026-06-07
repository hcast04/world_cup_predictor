from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW, DATA_PROCESSED, PROJECT_ROOT
from src.models.match_engine import MatchEngine
from src.models.elo_poisson import win_draw_loss_probabilities
import math


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"

def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def exact_score_probability(goals_a: int, goals_b: int, lambda_a: float, lambda_b: float) -> float:
    return poisson_pmf(goals_a, lambda_a) * poisson_pmf(goals_b, lambda_b)


def most_likely_score(lambda_a: float, lambda_b: float, max_goals: int = 6) -> tuple[int, int, float]:
    best_goals_a = 0
    best_goals_b = 0
    best_prob = -1.0

    for goals_a in range(max_goals + 1):
        for goals_b in range(max_goals + 1):
            prob = exact_score_probability(goals_a, goals_b, lambda_a, lambda_b)

            if prob > best_prob:
                best_goals_a = goals_a
                best_goals_b = goals_b
                best_prob = prob

    return best_goals_a, best_goals_b, best_prob


def result_from_score(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_b > goals_a:
        return "team_b_win"
    return "draw"


def probability_for_result(result: str, probs: dict[str, float]) -> float:
    if result == "team_a_win":
        return probs["team_a_win"]
    if result == "team_b_win":
        return probs["team_b_win"]
    if result == "draw":
        return probs["draw"]
    raise ValueError(f"Unknown result: {result}")


def joker_cost_for_stage(stage: str) -> int:
    if stage == "group":
        return 1

    if stage in {"round_of_32", "round_of_16", "quarter_final"}:
        return 2

    if stage in {"semi_final", "third_place", "final"}:
        return 3

    raise ValueError(f"Unknown stage: {stage}")


def round_multiplier_for_stage(stage: str) -> int:
    if stage == "group":
        return 1

    if stage in {"round_of_32", "round_of_16", "quarter_final"}:
        return 2

    if stage in {"semi_final", "third_place", "final"}:
        return 3

    raise ValueError(f"Unknown stage: {stage}")


def make_group_stage_predictions() -> pd.DataFrame:
    fixtures_path = DATA_RAW / "fixtures_2026.csv"
    elo_path = DATA_PROCESSED / "elo_ratings_model.csv"

    fixtures = pd.read_csv(fixtures_path)
    elo = pd.read_csv(elo_path)

    elo_lookup = dict(zip(elo["team"], elo["elo"]))

    # Add host lookup if you want later. For now, all fixtures have TBD venues.
    host_lookup = {}

    match_engine = MatchEngine(
        model_type="elo_poisson",
        elo_lookup=elo_lookup,
        host_lookup=host_lookup,
    )

    rows = []

    for _, row in fixtures.iterrows():
        stage = row["stage"]
        team_a = row["team_a"]
        team_b = row["team_b"]

        if not match_engine.has_team(team_a) or not match_engine.has_team(team_b):
            continue

        lambda_a, lambda_b = match_engine.expected_goals(team_a, team_b)
        probs = win_draw_loss_probabilities(lambda_a, lambda_b)

        best_score = None
        best_score_expected_points = -1.0
        best_score_exact_prob = 0.0
        best_score_result = None
        best_score_result_prob = 0.0

        for goals_a in range(7):
            for goals_b in range(7):
                score_result = result_from_score(goals_a, goals_b)
                score_exact_prob = exact_score_probability(
                    goals_a,
                    goals_b,
                    lambda_a,
                    lambda_b,
                )
                score_result_prob = probability_for_result(score_result, probs)

                score_expected_points = (
                    3.0 * score_result_prob
                    + 2.0 * score_exact_prob
                )

                if score_expected_points > best_score_expected_points:
                    best_score = (goals_a, goals_b)
                    best_score_expected_points = score_expected_points
                    best_score_exact_prob = score_exact_prob
                    best_score_result = score_result
                    best_score_result_prob = score_result_prob

        pred_goals_a, pred_goals_b = best_score
        predicted_result = best_score_result
        prob_exact = best_score_exact_prob
        prob_1x2 = best_score_result_prob

        round_multiplier = round_multiplier_for_stage(stage)
        joker_cost = joker_cost_for_stage(stage)

        expected_base_points = round_multiplier * (
            3.0 * prob_1x2
            + 2.0 * prob_exact
        )

        # Joker doubles the score, so the marginal gain equals expected_base_points.
        expected_joker_gain = expected_base_points
        joker_efficiency = expected_joker_gain / joker_cost

        rows.append(
            {
                "match_id": row["match_id"],
                "stage": stage,
                "group": row.get("group", ""),
                "date": row["date"],
                "team_a": team_a,
                "team_b": team_b,
                "effective_elo_a": match_engine.effective_elo(team_a),
                "effective_elo_b": match_engine.effective_elo(team_b),
                "lambda_a": lambda_a,
                "lambda_b": lambda_b,
                "prob_team_a_win": probs["team_a_win"],
                "prob_draw": probs["draw"],
                "prob_team_b_win": probs["team_b_win"],
                "predicted_result": predicted_result,
                "predicted_team_a_goals": pred_goals_a,
                "predicted_team_b_goals": pred_goals_b,
                "predicted_score": f"{pred_goals_a}-{pred_goals_b}",
                "prob_1x2_correct": prob_1x2,
                "prob_exact_score": prob_exact,
                "round_multiplier": round_multiplier,
                "joker_cost": joker_cost,
                "expected_base_points": expected_base_points,
                "expected_joker_gain": expected_joker_gain,
                "joker_efficiency": joker_efficiency,
            }
        )

    predictions = pd.DataFrame(rows)

    # Recommend group-stage jokers first: top 10 by expected value.
    predictions["recommended_group_joker"] = False
    group_mask = predictions["stage"].eq("group")

    top_group_jokers = (
        predictions[group_mask]
        .sort_values("joker_efficiency", ascending=False)
        .head(10)
        .index
    )

    predictions.loc[top_group_jokers, "recommended_group_joker"] = True

    return predictions


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = make_group_stage_predictions()

    predictions_path = OUTPUT_DIR / "stage1_match_predictions_elo_poisson.csv"
    jokers_path = OUTPUT_DIR / "stage1_group_joker_recommendations_elo_poisson.csv"

    predictions.to_csv(predictions_path, index=False)

    jokers = (
        predictions[predictions["recommended_group_joker"]]
        .sort_values("joker_efficiency", ascending=False)
        .copy()
    )
    jokers.to_csv(jokers_path, index=False)

    print(f"Saved match predictions to {predictions_path}")
    print(f"Saved joker recommendations to {jokers_path}")

    display_cols = [
        "match_id",
        "group",
        "team_a",
        "team_b",
        "predicted_score",
        "predicted_result",
        "prob_1x2_correct",
        "prob_exact_score",
        "expected_base_points",
        "joker_efficiency",
    ]

    print("\nTop group-stage joker candidates")
    print("--------------------------------")
    print(jokers[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()