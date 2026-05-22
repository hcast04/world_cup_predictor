from collections import defaultdict

import numpy as np
import pandas as pd

from src.data.loaders import PROJECT_ROOT
from src.simulation.simulate_group_from_probabilities import (
    simulate_group_stage_from_fixture_probabilities,
)
from src.simulation.third_place import rank_third_placed_teams


def main(n_simulations: int = 10_000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    input_path = (
        PROJECT_ROOT
        / "outputs"
        / "predictions"
        / "recent_form_fixture_predictions_2026.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run python scripts/predict_2026_fixtures_recent_form.py first."
        )

    fixture_predictions = pd.read_csv(input_path)

    position_counts = defaultdict(lambda: defaultdict(int))
    qualification_counts = defaultdict(int)
    top_two_counts = defaultdict(int)
    best_third_counts = defaultdict(int)
    points_sum = defaultdict(float)

    for _ in range(n_simulations):
        group_tables = simulate_group_stage_from_fixture_probabilities(
            fixture_predictions=fixture_predictions,
            rng=rng,
        )

        third_place_ranking = rank_third_placed_teams(group_tables)

        best_third_teams = set(
            third_place_ranking.loc[
                third_place_ranking["qualifies_as_third"],
                "team",
            ]
        )

        for group_name, table in group_tables.items():
            for _, row in table.iterrows():
                team = row["team"]
                position = int(row["position"])

                qualifies_top_two = position <= 2
                qualifies_best_third = position == 3 and team in best_third_teams
                qualifies = qualifies_top_two or qualifies_best_third

                position_counts[team][position] += 1
                points_sum[team] += float(row["points"])

                if qualifies:
                    qualification_counts[team] += 1

                if qualifies_top_two:
                    top_two_counts[team] += 1

                if qualifies_best_third:
                    best_third_counts[team] += 1

    rows = []

    max_position = max(
        position
        for team_counts in position_counts.values()
        for position in team_counts.keys()
    )

    for team in sorted(position_counts.keys()):
        row = {
            "team": team,
            "expected_points": points_sum[team] / n_simulations,
        }

        for position in range(1, max_position + 1):
            row[f"finish_{position}_prob"] = (
                position_counts[team][position] / n_simulations
            )

        row["top_two_prob"] = top_two_counts[team] / n_simulations
        row["best_third_prob"] = best_third_counts[team] / n_simulations
        row["qualify_prob"] = qualification_counts[team] / n_simulations

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        by=["qualify_prob", "top_two_prob", "expected_points"],
        ascending=False,
    )

    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "predictions"
        / "recent_form_group_stage_probabilities.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print("\nRecent-form group-stage qualification probabilities")
    print("---------------------------------------------------")
    print(f"Simulations: {n_simulations:,}")
    print(summary.head(30).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()