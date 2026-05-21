import numpy as np

from src.data.loaders import load_baseline_data
from src.simulation.tournament import simulate_group_stage


def main(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    teams, elo, fixtures = load_baseline_data()

    elo_lookup = dict(zip(elo["team"], elo["elo"]))
    host_lookup = dict(zip(teams["team"], teams["is_host"]))

    group_tables, qualification_table = simulate_group_stage(
        fixtures=fixtures,
        elo_lookup=elo_lookup,
        host_lookup=host_lookup,
        rng=rng,
    )

    print("\nSimulated group tables")

    for group_name, table in group_tables.items():
        print(f"\nGroup {group_name}")
        print(table.to_string(index=False))

    print("\nQualification table")
    print(qualification_table.to_string(index=False))


if __name__ == "__main__":
    main()