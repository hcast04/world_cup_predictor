import pandas as pd


def rank_third_placed_teams(group_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Rank third-placed teams across all groups.

    Baseline tie-breakers:
    1. points
    2. goal difference
    3. goals for
    4. group
    5. team

    This is enough for the simulator baseline. Later we can implement the exact
    FIFA fair-play and drawing-of-lots fallback if needed.
    """
    rows = []

    for group_name, table in group_tables.items():
        third_place = table.loc[table["position"] == 3].copy()

        if third_place.empty:
            raise ValueError(f"Group {group_name} does not have a third-placed team.")

        row = third_place.iloc[0].to_dict()
        row["group"] = group_name
        rows.append(row)

    thirds = pd.DataFrame(rows)

    thirds = thirds.sort_values(
        by=["points", "goal_difference", "goals_for", "group", "team"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    thirds.insert(0, "third_place_rank", range(1, len(thirds) + 1))
    thirds["qualifies_as_third"] = thirds["third_place_rank"] <= 8

    return thirds