import unicodedata
import re
import pandas as pd

from src.data.loaders import DATA_RAW, DATA_PROCESSED, PROJECT_ROOT


def clean_name(name: str) -> str:
    """
    Basic name key used for matching squad players to club-stat players.
    Removes accents, punctuation variants, and lowercases.
    """
    value = str(name).strip().lower()
    value = re.sub(r"\([^)]*\)", "", value) 
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))

    replacements = {
        ".": "",
        "-": " ",
        "'": "",
        "’": "",
        "`": "",
        "´": "",
        "  ": " ",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return " ".join(value.split())


def main() -> None:
    squads_path = DATA_RAW / "world_cup_squads_2026.csv"
    stats_path = DATA_PROCESSED / "club_player_stats_normalized.csv"

    if not squads_path.exists():
        raise FileNotFoundError(
            f"Missing {squads_path}. Run python -m src.data.collect_world_cup_squads_wikipedia first."
        )

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing {stats_path}. Run python -m src.data.normalize_club_player_sources first."
        )

    squads = pd.read_csv(squads_path)
    stats = pd.read_csv(stats_path)

    required_squad_cols = {"team", "player", "position"}
    required_stats_cols = {"player", "league", "minutes", "goals", "xg", "source_file"}

    missing_squad_cols = required_squad_cols - set(squads.columns)
    missing_stats_cols = required_stats_cols - set(stats.columns)

    if missing_squad_cols:
        raise ValueError(f"Missing squad columns: {missing_squad_cols}")

    if missing_stats_cols:
        raise ValueError(f"Missing stat columns: {missing_stats_cols}")

    squads = squads.copy()
    stats = stats.copy()

    squads["player_key"] = squads["player"].map(clean_name)
    stats["player_key"] = stats["player"].map(clean_name)

    # Keep one best stat row per player name, independent of team.
    # This helps identify whether we have any stat source for that player at all.
    stats_best = (
        stats.sort_values(
            by=["player_key", "minutes", "xg"],
            ascending=[True, False, False],
        )
        .drop_duplicates(subset=["player_key"], keep="first")
        .copy()
    )

    stats_keep_cols = [
        "player_key",
        "player",
        "team",
        "club",
        "league",
        "minutes",
        "goals",
        "xg",
        "npxg",
        "assists",
        "source_file",
    ]

    stats_keep_cols = [col for col in stats_keep_cols if col in stats_best.columns]

    merged = squads.merge(
        stats_best[stats_keep_cols],
        on="player_key",
        how="left",
        suffixes=("_squad", "_stats"),
    )

        # Standardize post-merge column names.
    rename_map = {}

    if "player_squad" in merged.columns:
        rename_map["player_squad"] = "squad_player"
    elif "player" in merged.columns:
        rename_map["player"] = "squad_player"

    if "team_squad" in merged.columns:
        rename_map["team_squad"] = "squad_team"
    elif "team" in merged.columns:
        rename_map["team"] = "squad_team"

    if "player_stats" in merged.columns:
        rename_map["player_stats"] = "stats_player"

    if rename_map:
        merged = merged.rename(columns=rename_map)

    if "stats_player" not in merged.columns:
        raise ValueError(
            f"Expected stats_player column after merge. Available columns: {list(merged.columns)}"
        )

    if "squad_team" not in merged.columns:
        raise ValueError(
            f"Expected squad_team column after merge. Available columns: {list(merged.columns)}"
        )

    merged["has_player_stats"] = merged["stats_player"].notna().astype(int)

    missing = merged[merged["has_player_stats"] == 0].copy()

    summary = (
        merged.groupby("squad_team")
        .agg(
            squad_players=("player_key", "count"),
            players_with_stats=("has_player_stats", "sum"),
        )
        .reset_index()
    )

    summary["players_missing_stats"] = (
        summary["squad_players"] - summary["players_with_stats"]
    )

    summary["stat_coverage_rate"] = (
        summary["players_with_stats"] / summary["squad_players"]
    ).fillna(0)

    summary = summary.rename(columns={"squad_team": "team"})

    summary = summary.sort_values(
        by=["stat_coverage_rate", "players_missing_stats"],
        ascending=[True, False],
    )

    def is_attacking_position(position: str) -> bool:
        pos = str(position).upper().strip()

        attacking_positions = {"FW", "F", "MF", "M", "MF,FW", "FW,MF", "M,F", "F,M"}

        return pos in attacking_positions


    missing_attackers = missing[
        missing["position"].map(is_attacking_position)
    ].copy()

    output_dir = PROJECT_ROOT / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_output = output_dir / "squad_player_stat_matching_detail.csv"
    missing_output = output_dir / "missing_squad_player_stats.csv"
    missing_attackers_output = output_dir / "missing_attacking_squad_player_stats.csv"
    summary_output = output_dir / "squad_player_stat_coverage_by_team.csv"

    merged.to_csv(merged_output, index=False)
    missing.to_csv(missing_output, index=False)
    missing_attackers.to_csv(missing_attackers_output, index=False)
    summary.to_csv(summary_output, index=False)

    print("\nSquad player stat coverage")
    print("--------------------------")
    print(f"Squad rows: {len(squads):,}")
    print(f"Players with stats: {merged['has_player_stats'].sum():,}")
    print(f"Players missing stats: {len(missing):,}")
    print(f"Missing attacking players: {len(missing_attackers):,}")

    print("\nWorst teams by stat coverage:")
    print(
        summary.head(25).to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nMissing attacking players sample:")
    cols = [
        col
        for col in ["squad_team", "squad_player", "position", "club", "source", "notes"]
        if col in missing_attackers.columns
    ]

    if missing_attackers.empty:
        print("none")
    else:
        print(missing_attackers[cols].head(80).to_string(index=False))

    print("\nSaved:")
    print(merged_output)
    print(missing_output)
    print(missing_attackers_output)
    print(summary_output)


if __name__ == "__main__":
    main()