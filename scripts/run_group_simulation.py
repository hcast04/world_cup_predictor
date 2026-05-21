from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders import PROJECT_ROOT, load_baseline_data
from src.simulation.simulate_group import simulate_group_from_fixtures


def main(n_simulations: int = 10_000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    teams, elo, fixtures = load_baseline_data()

    elo_lookup = dict(zip(elo["team"], elo["elo"]))
    host_lookup = dict(zip(teams["team"], teams["is_host"]))

    group_name = "A"

    position_counts = defaultdict(lambda: defaultdict(int))
    points_sum = defaultdict(float)

    for _ in range(n_simulations):
        table = simulate_group_from_fixtures(
            group_name=group_name,
            fixtures=fixtures,
            elo_lookup=elo_lookup,
            host_lookup=host_lookup,
            rng=rng,
        )

        for _, row in table.iterrows():
            team = row["team"]
            position = int(row["position"])
            position_counts[team][position] += 1
            points_sum[team] += float(row["points"])

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

        row["top_2_prob"] = (
            position_counts[team][1] + position_counts[team][2]
        ) / n_simulations

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        by=["top_2_prob", "finish_1_prob", "expected_points"],
        ascending=False,
    )

    output_dir = PROJECT_ROOT / "outputs" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"group_{group_name.lower()}_simulation_summary.csv"
    summary.to_csv(output_path, index=False)

    print(f"\nGroup {group_name} simulation summary")
    print(f"Simulations: {n_simulations:,}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nSaved results to:")
    print(output_path)


if __name__ == "__main__":
    main()