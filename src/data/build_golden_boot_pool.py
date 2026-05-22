from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_PROCESSED, load_teams, DATA_RAW



ATTACKING_POSITION_PATTERNS = ["FW", "MF,FW", "FW,MF"]


def _is_attacking_candidate(position: str) -> bool:
    position = str(position)

    return any(pattern in position for pattern in ATTACKING_POSITION_PATTERNS)

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

    # Add columns that exist in the generated pool but not necessarily in the override file.
    optional_defaults = {
        "club": "manual",
        "league": "manual",
        "minutes": 0,
        "goals": 0,
        "xg": 0,
        "npxg": 0,
        "club_scoring_score_shrunk": 0,
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

    This is not the final official squad list. It is a candidate pool based on:
    - national team
    - attacking position
    - club scoring score
    - minutes played

    Later, this should be filtered with official World Cup squads.
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
    df = df[df["minutes"] >= min_minutes].copy()
    df = df[df["position"].map(_is_attacking_candidate)].copy()

    # Estimate expected minutes from club usage.
    # This is a proxy until final squads and projected XIs are available.
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

    df["scoring_weight_source"] = "club_player_stats_2025_2026"

    # Keep top candidates per team by scoring score.
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
            "club_scoring_score_shrunk": df["club_scoring_score_shrunk"],
        }
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