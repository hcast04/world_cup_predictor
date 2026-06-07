import numpy as np
import pandas as pd


def compute_player_scoring_rates(players: pd.DataFrame) -> pd.DataFrame:
    """
    Compute expected World Cup goals per match for each player.

    Preferred signal:
    - club_scoring_score_shrunk, already adjusted for:
        - xG/goals/shots
        - missing-xG fallback
        - minutes reliability
        - league strength, if added upstream

    Fallback signal:
    - xG/90 + goals/90 if club_scoring_score_shrunk is unavailable.
    """
    df = players.copy()

    if "club_scoring_score_shrunk" in df.columns:
        base_rate_per90 = pd.to_numeric(
            df["club_scoring_score_shrunk"],
            errors="coerce",
        ).fillna(0.0)
    else:
        base_rate_per90 = (
            0.65 * pd.to_numeric(df["xg_per90"], errors="coerce").fillna(0.0)
            + 0.35 * pd.to_numeric(df["goals_per90"], errors="coerce").fillna(0.0)
        )

    minutes_factor = (
        pd.to_numeric(df["expected_minutes_per_match"], errors="coerce").fillna(0.0)
        / 90.0
    ).clip(lower=0.0, upper=1.0)

    starter_probability = pd.to_numeric(
        df["starter_probability"],
        errors="coerce",
    ).fillna(0.5)

    starter_factor = (0.75 + 0.25 * starter_probability).clip(lower=0.75, upper=1.0)

    penalty_factor = (
        1.0
        + 0.12
        * pd.to_numeric(df["is_penalty_taker"], errors="coerce").fillna(0.0)
    )

    df["base_rate_per90"] = base_rate_per90

    df["expected_goals_per_match"] = (
        base_rate_per90
        * minutes_factor
        * starter_factor
        * penalty_factor
    )

    # Safety cap. Nobody should have an expected WC scoring rate that is absurdly high.
    # This prevents extreme club seasons in weaker leagues from dominating too much.
    df["expected_goals_per_match"] = df["expected_goals_per_match"].clip(
        lower=0.0,
        upper=1.10,
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

    Important:
    - Team tournament path is handled through n_matches.
    - A player on a team eliminated in the group stage usually gets only 3 matches.
    - A player on a finalist can get up to 7 matches.
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
                "n_matches": n_matches,
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

    If team_matches_samples comes from the tournament simulation, then team path is
    already accounted for. Do not manually remove players from weaker teams.
    """
    rng = np.random.default_rng(seed)
    players_with_rates = compute_player_scoring_rates(players)

    win_counts = {player: 0.0 for player in players_with_rates["player"]}
    top_3_counts = {player: 0.0 for player in players_with_rates["player"]}
    goals_sum = {player: 0.0 for player in players_with_rates["player"]}
    expected_goals_sum = {player: 0.0 for player in players_with_rates["player"]}
    matches_sum = {player: 0.0 for player in players_with_rates["player"]}

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
            top_3_counts[player] += 1.0

        for _, row in simulated.iterrows():
            player = row["player"]
            goals_sum[player] += row["simulated_goals"]
            expected_goals_sum[player] += row["expected_goals"]
            matches_sum[player] += row["n_matches"]

    rows = []

    for player in players_with_rates["player"]:
        player_row = players_with_rates[players_with_rates["player"] == player].iloc[0]

        rows.append(
            {
                "player": player,
                "team": player_row["team"],
                "base_rate_per90": player_row["base_rate_per90"],
                "expected_goals_per_match": player_row["expected_goals_per_match"],
                "avg_team_matches": matches_sum[player] / n_simulations,
                "expected_simulated_goals": goals_sum[player] / n_simulations,
                "mean_expected_goals": expected_goals_sum[player] / n_simulations,
                "golden_boot_prob": win_counts[player] / n_simulations,
                "top_3_scorer_prob": top_3_counts[player] / n_simulations,
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["golden_boot_prob", "expected_simulated_goals"],
        ascending=False,
    )