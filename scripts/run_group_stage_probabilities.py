from collections import defaultdict

import numpy as np
import pandas as pd

from src.data.loaders import PROJECT_ROOT, load_baseline_data, load_model_elo_ratings
from src.simulation.tournament import get_group_names, simulate_group_stage


def main(n_simulations: int = 10_000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    teams, _, fixtures = load_baseline_data()
    elo = load_model_elo_ratings()

    elo_lookup = dict(zip(elo["team"], elo["elo"]))
    host_lookup = dict(zip(teams["team"], teams["is_host"]))

    all_groups = set(get_group_names(fixtures))

    position_counts = defaultdict(lambda: defaultdict(int))
    qualification_counts = defaultdict(int)
    top_two_counts = defaultdict(int)
    best_third_counts = defaultdict(int)
    points_sum = defaultdict(float)
    simulated_groups_seen = set()

    for _ in range(n_simulations):
        group_tables, qualification_table = simulate_group_stage(
            fixtures=fixtures,
            elo_lookup=elo_lookup,
            host_lookup=host_lookup,
            rng=rng,
        )

        simulated_groups_seen.update(group_tables.keys())

        for _, row in qualification_table.iterrows():
            team = row["team"]
            position = int(row["position"])

            position_counts[team][position] += 1
            points_sum[team] += float(row["points"])

            if bool(row["qualifies"]):
                qualification_counts[team] += 1

            if bool(row["qualifies_top_two"]):
                top_two_counts[team] += 1

            if bool(row["qualifies_best_third"]):
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

    output_dir = PROJECT_ROOT / "outputs" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "group_stage_qualification_probabilities.csv"
    summary.to_csv(output_path, index=False)

    skipped_groups = sorted(all_groups - simulated_groups_seen)

    print("\nGroup-stage qualification probabilities")
    print(f"Simulations: {n_simulations:,}")
    print(f"Simulated groups: {sorted(simulated_groups_seen)}")
    print(f"Skipped groups: {skipped_groups if skipped_groups else 'none'}")

    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nSaved results to:")
    print(output_path)


if __name__ == "__main__":
    main()