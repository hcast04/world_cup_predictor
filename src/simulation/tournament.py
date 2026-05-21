import numpy as np
import pandas as pd

from src.simulation.simulate_group import simulate_group_from_fixtures
from src.simulation.third_place import rank_third_placed_teams


def get_group_names(fixtures: pd.DataFrame) -> list[str]:
    """
    Return sorted group names from group-stage fixtures.
    """
    group_fixtures = fixtures[fixtures["stage"] == "group"].copy()

    groups = sorted(
        group
        for group in group_fixtures["group"].dropna().unique()
        if str(group).strip() != ""
    )

    return groups


def group_has_simulatable_match(
    group_name: str,
    fixtures: pd.DataFrame,
    elo_lookup: dict[str, float],
) -> bool:
    """
    Check whether a group has at least one match where both teams are known
    and both teams have Elo ratings.
    """
    group_fixtures = fixtures[
        (fixtures["stage"] == "group") & (fixtures["group"] == group_name)
    ].copy()

    for _, row in group_fixtures.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]

        if team_a == "TBD" or team_b == "TBD":
            continue

        if team_a not in elo_lookup or team_b not in elo_lookup:
            continue

        return True

    return False


def simulate_group_stage(
    fixtures: pd.DataFrame,
    elo_lookup: dict[str, float],
    host_lookup: dict[str, int],
    rng: np.random.Generator | None = None,
    skip_incomplete_groups: bool = True,
    manual_results: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Simulate every available group and return:
    - dictionary of group tables
    - combined qualification table

    During data collection, some groups may contain only TBD teams.
    If skip_incomplete_groups=True, those groups are skipped instead of crashing.
    """
    rng = rng or np.random.default_rng()

    group_names = get_group_names(fixtures)

    if not group_names:
        raise ValueError("No group-stage fixtures found.")

    group_tables = {}

    for group_name in group_names:
        if skip_incomplete_groups and not group_has_simulatable_match(
            group_name=group_name,
            fixtures=fixtures,
            elo_lookup=elo_lookup,
        ):
            continue

        group_tables[group_name] = simulate_group_from_fixtures(
            group_name=group_name,
            fixtures=fixtures,
            elo_lookup=elo_lookup,
            host_lookup=host_lookup,
            rng=rng,
            manual_results=manual_results,
        )

    if not group_tables:
        raise ValueError("No groups had simulatable matches.")

    # Best-third logic only really makes sense once we have multiple complete groups.
    # For partial testing, rank whatever third-placed teams exist.
    groups_with_third_place = {
        group_name: table
        for group_name, table in group_tables.items()
        if (table["position"] == 3).any()
    }

    if groups_with_third_place:
        third_place_ranking = rank_third_placed_teams(groups_with_third_place)

        best_third_teams = set(
            third_place_ranking.loc[
                third_place_ranking["qualifies_as_third"],
                "team",
            ]
        )
    else:
        best_third_teams = set()

    qualification_rows = []

    for group_name, table in group_tables.items():
        for _, row in table.iterrows():
            position = int(row["position"])
            team = row["team"]

            qualifies_top_two = position <= 2
            qualifies_best_third = position == 3 and team in best_third_teams
            qualifies = qualifies_top_two or qualifies_best_third

            output_row = row.to_dict()
            output_row["group"] = group_name
            output_row["qualifies_top_two"] = qualifies_top_two
            output_row["qualifies_best_third"] = qualifies_best_third
            output_row["qualifies"] = qualifies

            qualification_rows.append(output_row)

    qualification_table = pd.DataFrame(qualification_rows)

    qualification_table = qualification_table.sort_values(
        by=["group", "position"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return group_tables, qualification_table