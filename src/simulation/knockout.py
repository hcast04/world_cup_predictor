import numpy as np
import pandas as pd

from src.data.loaders import DATA_PROCESSED
from src.models.elo_poisson import expected_goals_from_elo, win_draw_loss_probabilities
from src.simulation.match import simulate_knockout_winner


STAGE_ORDER = [
    "round_of_32",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "third_place",
    "final",
]


def load_squad_elo_adjustments() -> dict[str, float]:
    """
    Load squad-based Elo residual adjustments.

    If the file does not exist, return an empty dictionary so knockout
    predictions fall back to raw Elo.
    """
    path = DATA_PROCESSED / "team_squad_strength.csv"

    if not path.exists():
        return {}

    df = pd.read_csv(path)

    required = {"team", "squad_elo_adjustment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in team_squad_strength.csv: {missing}")

    return dict(
        zip(
            df["team"].astype(str),
            pd.to_numeric(df["squad_elo_adjustment"], errors="coerce").fillna(0.0),
        )
    )


def effective_elo(
    team: str,
    elo_lookup: dict[str, float],
    squad_elo_adjustments: dict[str, float],
) -> float:
    return float(elo_lookup[team]) + float(squad_elo_adjustments.get(team, 0.0))


def predict_knockout_fixture_probabilities(
    fixtures: pd.DataFrame,
    elo_lookup: dict[str, float],
    host_lookup: dict[str, int],
) -> pd.DataFrame:
    """
    Compute match probabilities for manually entered knockout fixtures.

    This does not advance a bracket automatically. It simply predicts each
    known fixture independently.

    Uses effective Elo:
        effective_elo = base_elo + squad_elo_adjustment
    """
    squad_elo_adjustments = load_squad_elo_adjustments()

    rows = []

    for _, row in fixtures.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]

        if team_a == "TBD" or team_b == "TBD":
            continue

        if team_a not in elo_lookup or team_b not in elo_lookup:
            continue

        elo_a = effective_elo(team_a, elo_lookup, squad_elo_adjustments)
        elo_b = effective_elo(team_b, elo_lookup, squad_elo_adjustments)

        lambda_a, lambda_b = expected_goals_from_elo(
            elo_a=elo_a,
            elo_b=elo_b,
            team_a_is_host=bool(host_lookup.get(team_a, 0)),
            team_b_is_host=bool(host_lookup.get(team_b, 0)),
        )

        probs_90 = win_draw_loss_probabilities(lambda_a, lambda_b)

        # Approximate advancement probability:
        # P(advance) = P(win in 90) + P(draw in 90) * P(win after ET/pens)
        elo_diff = elo_a - elo_b
        raw_p_a = 1.0 / (1.0 + np.exp(-elo_diff / 400.0))
        p_a_after_draw = 0.5 + 0.15 * (raw_p_a - 0.5)

        p_a_advance = probs_90["team_a_win"] + probs_90["draw"] * p_a_after_draw
        p_b_advance = 1.0 - p_a_advance

        rows.append(
            {
                "match_id": row["match_id"],
                "stage": row["stage"],
                "date": row["date"],
                "team_a": team_a,
                "team_b": team_b,
                "base_elo_a": elo_lookup[team_a],
                "base_elo_b": elo_lookup[team_b],
                "squad_elo_adjustment_a": squad_elo_adjustments.get(team_a, 0.0),
                "squad_elo_adjustment_b": squad_elo_adjustments.get(team_b, 0.0),
                "effective_elo_a": elo_a,
                "effective_elo_b": elo_b,
                "lambda_a": lambda_a,
                "lambda_b": lambda_b,
                "team_a_win_90_prob": probs_90["team_a_win"],
                "draw_90_prob": probs_90["draw"],
                "team_b_win_90_prob": probs_90["team_b_win"],
                "team_a_advance_prob": p_a_advance,
                "team_b_advance_prob": p_b_advance,
            }
        )

    return pd.DataFrame(rows)


def simulate_manual_knockout_stage(
    fixtures: pd.DataFrame,
    elo_lookup: dict[str, float],
    host_lookup: dict[str, int],
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Simulate currently known knockout fixtures.

    This is useful for Stage 2 and Stage 3 when you manually enter the fixtures.
    It returns one simulated winner per known fixture.

    Uses effective Elo:
        effective_elo = base_elo + squad_elo_adjustment
    """
    squad_elo_adjustments = load_squad_elo_adjustments()

    rng = rng or np.random.default_rng()
    rows = []

    for _, row in fixtures.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]

        if team_a == "TBD" or team_b == "TBD":
            continue

        if team_a not in elo_lookup or team_b not in elo_lookup:
            continue

        elo_a = effective_elo(team_a, elo_lookup, squad_elo_adjustments)
        elo_b = effective_elo(team_b, elo_lookup, squad_elo_adjustments)

        lambda_a, lambda_b = expected_goals_from_elo(
            elo_a=elo_a,
            elo_b=elo_b,
            team_a_is_host=bool(host_lookup.get(team_a, 0)),
            team_b_is_host=bool(host_lookup.get(team_b, 0)),
        )

        result = simulate_knockout_winner(
            team_a=team_a,
            team_b=team_b,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            elo_a=elo_a,
            elo_b=elo_b,
            rng=rng,
        )

        rows.append(
            {
                "match_id": row["match_id"],
                "stage": row["stage"],
                "team_a": team_a,
                "team_b": team_b,
                "base_elo_a": elo_lookup[team_a],
                "base_elo_b": elo_lookup[team_b],
                "squad_elo_adjustment_a": squad_elo_adjustments.get(team_a, 0.0),
                "squad_elo_adjustment_b": squad_elo_adjustments.get(team_b, 0.0),
                "effective_elo_a": elo_a,
                "effective_elo_b": elo_b,
                "goals_a_90": result.goals_a,
                "goals_b_90": result.goals_b,
                "winner": result.winner,
                "decided_after_draw": result.is_draw,
            }
        )

    return pd.DataFrame(rows)