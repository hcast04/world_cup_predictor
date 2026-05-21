from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW
from src.data.load_groups import load_groups


GROUP_STAGE_PAIRINGS = [
    (1, 2),
    (3, 4),
    (1, 3),
    (4, 2),
    (4, 1),
    (2, 3),
]


def generate_group_stage_fixtures(
    groups: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Generate group-stage fixtures from group/slot/team assignments.

    This creates structurally correct group matches. Dates and venues are left
    as TBD for now and can be merged with official fixture metadata later.
    """
    groups = groups if groups is not None else load_groups()
    output_path = output_path or DATA_RAW / "fixtures_2026.csv"

    rows = []
    match_id = 1

    for group_name in sorted(groups["group"].unique()):
        group_df = groups[groups["group"] == group_name].copy()
        slot_to_team = dict(zip(group_df["slot"], group_df["team"]))

        for slot_a, slot_b in GROUP_STAGE_PAIRINGS:
            team_a = slot_to_team.get(slot_a, "TBD")
            team_b = slot_to_team.get(slot_b, "TBD")

            rows.append(
                {
                    "match_id": match_id,
                    "stage": "group",
                    "group": group_name,
                    "date": "2026-06-01",
                    "team_a": team_a,
                    "team_b": team_b,
                    "venue": "TBD",
                    "city": "TBD",
                    "country": "TBD",
                }
            )

            match_id += 1

    fixtures = pd.DataFrame(rows)
    fixtures.to_csv(output_path, index=False)

    return fixtures


if __name__ == "__main__":
    fixtures = generate_group_stage_fixtures()
    print(fixtures.to_string(index=False))
    print(f"\nSaved fixtures to {DATA_RAW / 'fixtures_2026.csv'}")