import numpy as np
import pandas as pd


def compute_player_scoring_rates(players: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a baseline expected goals per match for each player.

    This is a first simple model:
    - xG/90 is the main signal
    - goals/90 is secondary
    - expected minutes scale the rate
    - penalty takers get a small boost
    """
    df = players.copy()

    base_rate_per90 = 0.65 * df["xg_per90"] + 0.35 * df["goals_per90"]

    minutes_factor = df["expected_minutes_per_match"] / 90.0
    starter_factor = 0.75 + 0.25 * df["starter_probability"]
    penalty_factor = 1.0 + 0.12 * df["is_penalty_taker"]

    df["expected_goals_per_match"] = (
        base_rate_per90 * minutes_factor * starter_factor * penalty_factor
    )

    return df


def simulate_player_goals(
    players_with_rates: pd.DataFrame,
    team_matches: dict[str, int],
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Simulate tournament goals for each player.

    team_matches maps each team to the number of matches it plays in the simulated tournament.
    """
    rng = rng or np.random.default_rng()

    rows = []

    for _, row in players_with_rates.iterrows():
        team = row["team"]
        n_matches = int(team_matches.get(team, 0))

        expected_goals = row["expected_goals_per_match"] * n_matches
        simulated_goals = int(rng.poisson(expected_goals))

        rows.append(
            {
                "player": row["player"],
                "team": team,
                "expected_goals": expected_goals,
                "simulated_goals": simulated_goals,
            }
        )

    return pd.DataFrame(rows)


def simulate_golden_boot(
    players: pd.DataFrame,
    team_matches_samples: list[dict[str, int]],
    n_simulations: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Estimate Golden Boot probabilities.

    team_matches_samples is a list where each element maps:
        team -> number of matches played

    For now, this can come from a fake or group-stage-only assumption.
    Later, it should come from tournament simulations.
    """
    rng = np.random.default_rng(seed)
    players_with_rates = compute_player_scoring_rates(players)

    win_counts = {player: 0 for player in players_with_rates["player"]}
    top_3_counts = {player: 0 for player in players_with_rates["player"]}
    goals_sum = {player: 0.0 for player in players_with_rates["player"]}

    for i in range(n_simulations):
        team_matches = team_matches_samples[i % len(team_matches_samples)]

        simulated = simulate_player_goals(
            players_with_rates=players_with_rates,
            team_matches=team_matches,
            rng=rng,
        )

        simulated = simulated.sort_values(
            by=["simulated_goals", "expected_goals", "player"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        max_goals = simulated.loc[0, "simulated_goals"]
        winners = simulated[simulated["simulated_goals"] == max_goals]["player"].tolist()

        # Split Golden Boot probability across tied top scorers.
        for winner in winners:
            win_counts[winner] += 1.0 / len(winners)

        for player in simulated.head(3)["player"]:
            top_3_counts[player] += 1

        for _, row in simulated.iterrows():
            goals_sum[row["player"]] += row["simulated_goals"]

    rows = []

    for player in players_with_rates["player"]:
        player_row = players_with_rates[players_with_rates["player"] == player].iloc[0]

        rows.append(
            {
                "player": player,
                "team": player_row["team"],
                "expected_goals_per_match": player_row["expected_goals_per_match"],
                "expected_simulated_goals": goals_sum[player] / n_simulations,
                "golden_boot_prob": win_counts[player] / n_simulations,
                "top_3_scorer_prob": top_3_counts[player] / n_simulations,
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["golden_boot_prob", "expected_simulated_goals"],
        ascending=False,
    )