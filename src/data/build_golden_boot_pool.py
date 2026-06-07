from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_PROCESSED, DATA_RAW, load_teams


ATTACKING_POSITIONS = {
    "FW",
    "MF",
    "FW,MF",
    "MF,FW",
    "F",
    "M",
    "F,M",
    "M,F",
}


def _clean_player_key(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(".", "")
        .replace("-", " ")
        .replace("'", "")
        .replace("’", "")
    )


def _is_attacking_candidate(position: str) -> bool:
    position = str(position).upper().strip()
    return position in ATTACKING_POSITIONS


def _load_manual_overrides() -> pd.DataFrame:
    path = DATA_RAW / "golden_boot_manual_overrides.csv"

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if df.empty:
        return pd.DataFrame()

    required_columns = {
        "player",
        "team",
        "position",
        "expected_minutes_per_match",
        "starter_probability",
        "goals_per90",
        "xg_per90",
        "is_penalty_taker",
        "scoring_weight_source",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in golden_boot_manual_overrides.csv: {missing}")

    optional_defaults = {
        "club": "manual",
        "league": "manual",
        "minutes": 0,
        "goals": 0,
        "xg": 0,
        "npxg": 0,
        "shots": 0,
        "shots_per90": 0,
        "has_xg_data": False,
        "scoring_signal_type": "manual",
        "club_scoring_score": 0,
        "club_scoring_score_shrunk": 0,
        "source_file": "golden_boot_manual_overrides.csv",
    }

    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default

    return df


def build_golden_boot_pool(
    input_path: Path | None = None,
    output_path: Path | None = None,
    min_minutes: int = 180,
    max_players_per_team: int = 8,
) -> pd.DataFrame:
    """
    Build a Golden Boot candidate pool from club player scoring features.

    Input:
    - data/processed/player_scoring_features.csv

    Output:
    - data/processed/golden_boot_player_pool.csv

    Notes:
    - Restricts to World Cup teams.
    - If in_world_cup_squad exists, restricts to squad players.
    - Keeps attacking candidates only.
    - Keeps useful audit/provenance columns such as shots, source_file,
      has_xg_data, and scoring_signal_type.
    """
    input_path = input_path or DATA_PROCESSED / "player_scoring_features.csv"
    output_path = output_path or DATA_PROCESSED / "golden_boot_player_pool.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run python -m src.data.build_player_scoring_features first."
        )

    scoring = pd.read_csv(input_path)
    teams = load_teams()

    world_cup_teams = set(teams["team"])

    df = scoring[scoring["team"].isin(world_cup_teams)].copy()

    if "in_world_cup_squad" in df.columns:
        df = df[df["in_world_cup_squad"] == 1].copy()

    required_columns = {
        "player",
        "team",
        "position",
        "starts_per_match",
        "minutes_reliability",
        "is_penalty_taker_proxy",
        "goals_per90",
        "xg_per90",
        "club",
        "league",
        "minutes",
        "goals",
        "xg",
        "npxg",
        "club_scoring_score_shrunk",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected player scoring columns: {missing}")

    optional_defaults = {
        "shots": 0,
        "shots_per90": 0,
        "has_xg_data": pd.NA,
        "scoring_signal_type": "unknown",
        "club_scoring_score": pd.NA,
        "source_file": pd.NA,
    }

    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default

    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
    df["club_scoring_score_shrunk"] = pd.to_numeric(
        df["club_scoring_score_shrunk"], errors="coerce"
    ).fillna(0)

    df = df[df["minutes"] >= min_minutes].copy()
    df = df[df["position"].map(_is_attacking_candidate)].copy()

    df["expected_minutes_per_match"] = (
        35
        + 45 * df["starts_per_match"].clip(lower=0, upper=1)
        + 10 * df["minutes_reliability"].clip(lower=0, upper=1)
    ).clip(lower=20, upper=90)

    df["starter_probability"] = (
        0.25
        + 0.65 * df["starts_per_match"].clip(lower=0, upper=1)
        + 0.10 * df["minutes_reliability"].clip(lower=0, upper=1)
    ).clip(lower=0.05, upper=0.98)

    df["is_penalty_taker"] = df["is_penalty_taker_proxy"].astype(int)
    df["scoring_weight_source"] = "club_player_stats"

    df = df.sort_values(
        by=["team", "club_scoring_score_shrunk", "minutes"],
        ascending=[True, False, False],
    )

    df = df.groupby("team", as_index=False).head(max_players_per_team).copy()

    out = pd.DataFrame(
        {
            "player": df["player"],
            "team": df["team"],
            "position": df["position"],
            "expected_minutes_per_match": df["expected_minutes_per_match"],
            "starter_probability": df["starter_probability"],
            "goals_per90": df["goals_per90"],
            "xg_per90": df["xg_per90"],
            "is_penalty_taker": df["is_penalty_taker"],
            "scoring_weight_source": df["scoring_weight_source"],
            "club": df["club"],
            "league": df["league"],
            "minutes": df["minutes"],
            "goals": df["goals"],
            "xg": df["xg"],
            "npxg": df["npxg"],
            "shots": df["shots"],
            "shots_per90": df["shots_per90"],
            "has_xg_data": df["has_xg_data"],
            "scoring_signal_type": df["scoring_signal_type"],
            "club_scoring_score": df["club_scoring_score"],
            "club_scoring_score_shrunk": df["club_scoring_score_shrunk"],
            "source_file": df["source_file"],
        }
    )

    out["player_key"] = out["player"].map(_clean_player_key)

    out = (
        out.sort_values(
            by=["team", "player_key", "minutes", "club_scoring_score_shrunk"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(subset=["team", "player_key"], keep="first")
        .drop(columns=["player_key"])
        .reset_index(drop=True)
    )

    manual_overrides = _load_manual_overrides()

    if not manual_overrides.empty:
        override_cols = list(out.columns)

        for col in override_cols:
            if col not in manual_overrides.columns:
                manual_overrides[col] = pd.NA

        manual_overrides = manual_overrides[override_cols]

        out = pd.concat([out, manual_overrides], ignore_index=True)

        source_priority = (
            out["scoring_weight_source"]
            .astype(str)
            .str.contains("manual", case=False, na=False)
            .astype(int)
        )

        out = (
            out.assign(source_priority=source_priority)
            .sort_values(
                ["player", "team", "source_priority"],
                ascending=[True, True, False],
            )
            .assign(player_key=lambda x: x["player"].map(_clean_player_key))
            .drop_duplicates(subset=["team", "player_key"], keep="first")
            .drop(columns=["player_key", "source_priority"])
        )

    out = out.sort_values(
        by=["team", "club_scoring_score_shrunk"],
        ascending=[True, False],
    ).reset_index(drop=True)

    out.to_csv(output_path, index=False)

    return out


if __name__ == "__main__":
    pool = build_golden_boot_pool()
    print(pool.head(50).to_string(index=False))
    print(f"\nSaved {len(pool):,} rows to {DATA_PROCESSED / 'golden_boot_player_pool.csv'}")