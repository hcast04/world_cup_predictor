from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW, DATA_PROCESSED


def _clean_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(".", "")
        .replace("-", " ")
        .replace("'", "")
        .replace("’", "")
        .replace("é", "e")
        .replace("è", "e")
        .replace("á", "a")
        .replace("à", "a")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )


def _clean_player_key(name: str) -> str:
    return _clean_name(name)


def apply_manual_squad_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply manual player -> national team mappings.

    This fixes cases where club-stat rows have good club data but missing/ambiguous
    nationality/team fields, e.g. Kane, Mbappé, Vinícius, Messi, Ronaldo.
    """
    path = DATA_RAW / "squad_player_manual_mappings.csv"

    if not path.exists():
        return df

    mappings = pd.read_csv(path)

    required = {"player", "team", "squad_player", "squad_team"}
    missing = required - set(mappings.columns)
    if missing:
        raise ValueError(f"Missing columns in squad_player_manual_mappings.csv: {missing}")

    df = df.copy()
    df["_player_key"] = df["player"].map(_clean_player_key)

    mappings = mappings.copy()
    mappings["_player_key"] = mappings["player"].map(_clean_player_key)

    mapping_by_key = mappings.drop_duplicates("_player_key").set_index("_player_key")

    manual_match_count = 0

    for player_key, mapping_row in mapping_by_key.iterrows():
        mask = df["_player_key"].eq(player_key)

        if not mask.any():
            continue

        manual_match_count += int(mask.sum())

        df.loc[mask, "team"] = mapping_row["team"]
        df.loc[mask, "nation_raw"] = mapping_row["team"]
        df.loc[mask, "in_world_cup_squad"] = 1
        df.loc[mask, "squad_player"] = mapping_row["squad_player"]
        df.loc[mask, "squad_team"] = mapping_row["squad_team"]

    df = df.drop(columns=["_player_key"])

    print(f"Manual squad mapping rows applied: {manual_match_count:,}")

    return df


def apply_squad_player_mapping(
    player_stats_path: Path | None = None,
    squads_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Use World Cup squad data as the authoritative player -> national team mapping.

    Input:
    - data/processed/club_player_stats_normalized.csv
    - data/raw/world_cup_squads_2026.csv
    - optionally data/raw/squad_player_manual_mappings.csv

    Output:
    - data/processed/club_player_stats_with_squads.csv
    """
    player_stats_path = player_stats_path or DATA_PROCESSED / "club_player_stats_normalized.csv"
    squads_path = squads_path or DATA_RAW / "world_cup_squads_2026.csv"
    output_path = output_path or DATA_PROCESSED / "club_player_stats_with_squads.csv"

    if not player_stats_path.exists():
        raise FileNotFoundError(
            f"Missing {player_stats_path}. Run python -m src.data.normalize_club_player_sources first."
        )

    if not squads_path.exists():
        raise FileNotFoundError(
            f"Missing {squads_path}. Create data/raw/world_cup_squads_2026.csv first."
        )

    stats = pd.read_csv(player_stats_path)
    squads = pd.read_csv(squads_path)

    required_squad_columns = {"team", "player"}
    missing = required_squad_columns - set(squads.columns)

    if missing:
        raise ValueError(f"Missing columns in world_cup_squads_2026.csv: {missing}")

    stats["player_key"] = stats["player"].map(_clean_name)
    squads["player_key"] = squads["player"].map(_clean_name)

    squads_for_join = squads.copy()
    squads_for_join = squads_for_join.rename(
        columns={
            "team": "squad_team",
            "player": "squad_player",
            "position": "squad_position",
            "club": "squad_club",
        }
    )

    keep_cols = [
        col
        for col in [
            "player_key",
            "squad_team",
            "squad_player",
            "squad_position",
            "squad_club",
            "source",
            "notes",
        ]
        if col in squads_for_join.columns
    ]

    merged = stats.merge(
        squads_for_join[keep_cols],
        on="player_key",
        how="left",
    )

    matched = merged["squad_team"].notna()

    # For players found in the squad list, overwrite team/nation with squad team.
    merged.loc[matched, "team"] = merged.loc[matched, "squad_team"]
    merged.loc[matched, "nation_raw"] = merged.loc[matched, "squad_team"]

    # Prefer squad position if available.
    if "squad_position" in merged.columns:
        has_squad_position = merged["squad_position"].notna() & (
            merged["squad_position"].astype(str).str.len() > 0
        )
        merged.loc[has_squad_position, "position"] = merged.loc[
            has_squad_position, "squad_position"
        ]

    merged["in_world_cup_squad"] = matched.astype(int)

    merged = merged.drop(columns=["player_key"])

    # IMPORTANT: apply manual mappings after automatic matching.
    # This fixes elite players whose best club-stat row has missing team/nation.
    merged = apply_manual_squad_mappings(merged)

    final_matched = merged["in_world_cup_squad"].eq(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print("\nSquad mapping summary")
    print("---------------------")
    print(f"Club stat rows: {len(stats):,}")
    print(f"Squad rows: {len(squads):,}")
    print(f"Automatically matched club stat rows: {matched.sum():,}")
    print(f"Final matched club stat rows: {final_matched.sum():,}")
    print(f"Unique matched squad players: {merged.loc[final_matched, 'player'].nunique():,}")

    return merged


if __name__ == "__main__":
    out = apply_squad_player_mapping()
    print(f"\nSaved to {DATA_PROCESSED / 'club_player_stats_with_squads.csv'}")