import numpy as np
import pandas as pd


def fit_team_goal_strengths(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Fit simple team attack/defence strengths from historical match results.

    This is a baseline model:
    - attack_strength = team's average goals scored per match
    - defence_strength = team's average goals conceded per match
    - both home/team_a and away/team_b appearances are included

    Output columns:
    - team
    - matches
    - goals_for_per_match
    - goals_against_per_match
    - attack_index
    - defence_index

    attack_index > 1 means better than average attack.
    defence_index < 1 means better than average defence.
    """
    team_a = matches.rename(
        columns={
            "team_a": "team",
            "goals_a": "goals_for",
            "goals_b": "goals_against",
        }
    )[["team", "goals_for", "goals_against"]]

    team_b = matches.rename(
        columns={
            "team_b": "team",
            "goals_b": "goals_for",
            "goals_a": "goals_against",
        }
    )[["team", "goals_for", "goals_against"]]

    long = pd.concat([team_a, team_b], ignore_index=True)

    team_stats = (
        long.groupby("team")
        .agg(
            matches=("team", "size"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
        )
        .reset_index()
    )

    team_stats["goals_for_per_match"] = team_stats["goals_for"] / team_stats["matches"]
    team_stats["goals_against_per_match"] = (
        team_stats["goals_against"] / team_stats["matches"]
    )

    global_goals_for = long["goals_for"].mean()
    global_goals_against = long["goals_against"].mean()

    team_stats["attack_index"] = team_stats["goals_for_per_match"] / global_goals_for
    team_stats["defence_index"] = (
        team_stats["goals_against_per_match"] / global_goals_against
    )

    return team_stats.sort_values("team").reset_index(drop=True)


def predict_expected_goals_strength_baseline(
    team_a: str,
    team_b: str,
    strengths: pd.DataFrame,
    base_goals: float = 1.35,
    shrinkage_matches: int = 20,
) -> tuple[float, float]:
    """
    Predict expected goals from simple attack/defence strengths.

    Uses shrinkage so small-sample teams are pulled toward average.
    """
    strengths_lookup = strengths.set_index("team").to_dict(orient="index")

    def get_strength(team: str) -> dict:
        if team not in strengths_lookup:
            return {
                "matches": 0,
                "attack_index": 1.0,
                "defence_index": 1.0,
            }
        return strengths_lookup[team]

    a = get_strength(team_a)
    b = get_strength(team_b)

    def shrink(value: float, n_matches: int) -> float:
        weight = n_matches / (n_matches + shrinkage_matches)
        return weight * value + (1.0 - weight) * 1.0

    attack_a = shrink(a["attack_index"], a["matches"])
    defence_a = shrink(a["defence_index"], a["matches"])
    attack_b = shrink(b["attack_index"], b["matches"])
    defence_b = shrink(b["defence_index"], b["matches"])

    lambda_a = base_goals * attack_a * defence_b
    lambda_b = base_goals * attack_b * defence_a

    # Avoid pathological tiny/huge values.
    lambda_a = float(np.clip(lambda_a, 0.15, 5.0))
    lambda_b = float(np.clip(lambda_b, 0.15, 5.0))

    return lambda_a, lambda_b