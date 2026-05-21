import numpy as np

from src.data.loaders import load_baseline_data
from src.models.elo_poisson import expected_goals_from_elo, win_draw_loss_probabilities
from src.simulation.match import simulate_scoreline


def main(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    teams, elo, fixtures = load_baseline_data()

    elo_lookup = dict(zip(elo["team"], elo["elo"]))
    host_lookup = dict(zip(teams["team"], teams["is_host"]))

    print("\nLoaded data")
    print(f"- Teams: {len(teams)}")
    print(f"- Elo ratings: {len(elo)}")
    print(f"- Fixtures: {len(fixtures)}")

    print("\nPlaceholder match simulations")

    for _, row in fixtures.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]

        if team_a == "TBD" or team_b == "TBD":
            print(f"- Match {row['match_id']}: {team_a} vs {team_b} skipped because one team is TBD.")
            continue

        if team_a not in elo_lookup or team_b not in elo_lookup:
            print(f"- Match {row['match_id']}: {team_a} vs {team_b} skipped because Elo is missing.")
            continue

        lambda_a, lambda_b = expected_goals_from_elo(
            elo_a=elo_lookup[team_a],
            elo_b=elo_lookup[team_b],
            team_a_is_host=bool(host_lookup.get(team_a, 0)),
            team_b_is_host=bool(host_lookup.get(team_b, 0)),
        )

        probs = win_draw_loss_probabilities(lambda_a, lambda_b)

        result = simulate_scoreline(
            team_a=team_a,
            team_b=team_b,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            rng=rng,
        )

        print(
            f"- Match {row['match_id']}: {team_a} vs {team_b} | "
            f"xG {lambda_a:.2f}-{lambda_b:.2f} | "
            f"P(W/D/L) {probs['team_a_win']:.2%}/"
            f"{probs['draw']:.2%}/"
            f"{probs['team_b_win']:.2%} | "
            f"simulated score {result.goals_a}-{result.goals_b}"
        )


if __name__ == "__main__":
    main()