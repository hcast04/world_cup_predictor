import pandas as pd

from src.data.loaders import PROJECT_ROOT


HIGH_PRIORITY_TEAMS = {
    "Qatar",
    "Iran",
    "South Africa",
    "Australia",
    "Ecuador",
    "New Zealand",
    "Saudi Arabia",
    "Iraq",
    "Jordan",
    "Uzbekistan",
    "Panama",
}


def main() -> None:
    path = PROJECT_ROOT / "outputs" / "tables" / "missing_attacking_squad_player_stats.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python scripts/audit_missing_squad_player_stats.py first."
        )

    df = pd.read_csv(path)

    if df.empty:
        print("No missing attacking players.")
        return

    df["priority"] = "medium"

    df.loc[df["squad_team"].isin(HIGH_PRIORITY_TEAMS), "priority"] = "high"

    df.loc[
        df["position"].astype(str).str.upper().isin(["FW", "F"]),
        "position_priority",
    ] = 1

    df["position_priority"] = df["position_priority"].fillna(2)

    df = df.sort_values(
        by=["priority", "squad_team", "position_priority", "squad_player"],
        ascending=[True, True, True, True],
    )

    # Put high first.
    priority_order = {"high": 0, "medium": 1, "low": 2}
    df["priority_order"] = df["priority"].map(priority_order).fillna(1)

    df = df.sort_values(
        by=["priority_order", "squad_team", "position_priority", "squad_player"],
        ascending=[True, True, True, True],
    ).drop(columns=["priority_order"])

    output_path = PROJECT_ROOT / "outputs" / "tables" / "missing_attacking_stats_priority_list.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nMissing attacking player stat priority list")
    print("-------------------------------------------")
    print(f"Rows: {len(df):,}")

    print("\nHigh-priority missing players:")
    high = df[df["priority"] == "high"]

    if high.empty:
        print("none")
    else:
        print(
            high[
                [
                    "squad_team",
                    "squad_player",
                    "position",
                    "priority",
                    "source",
                ]
            ]
            .head(120)
            .to_string(index=False)
        )

    print(f"\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()