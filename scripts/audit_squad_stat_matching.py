import pandas as pd

from src.data.loaders import DATA_PROCESSED, PROJECT_ROOT


def main() -> None:
    path = DATA_PROCESSED / "club_player_stats_with_squads.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m src.data.apply_squad_player_mapping first."
        )

    df = pd.read_csv(path)

    if "in_world_cup_squad" not in df.columns:
        raise ValueError("Missing in_world_cup_squad column. Check apply_squad_player_mapping.py.")

    matched = df[df["in_world_cup_squad"] == 1].copy()

    summary = (
        matched.groupby("team")
        .agg(
            matched_stat_rows=("player", "count"),
            unique_matched_players=("player", "nunique"),
            attacking_players=(
                "position",
                lambda s: s.astype(str).str.upper().str.contains("F").sum(),
            ),
            total_minutes=("minutes", "sum"),
            total_xg=("xg", "sum"),
        )
        .reset_index()
        .sort_values("unique_matched_players")
    )

    output_path = PROJECT_ROOT / "outputs" / "tables" / "squad_stat_matching_audit.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print("\nSquad-to-stat matching audit")
    print("----------------------------")
    print(summary.to_string(index=False))

    print(f"\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()