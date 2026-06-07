from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_PROCESSED


def _safe_numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index)

    return pd.to_numeric(df[column], errors="coerce").fillna(default)

def _league_strength_multiplier(row: pd.Series) -> float:
    """
    Approximate adjustment for club scoring difficulty.

    This prevents raw goals from MLS/Saudi/Liga MX/Belgian league/etc.
    being treated as directly equivalent to goals/xG from top European leagues.
    """
    league = str(row.get("league", "")).lower()
    source_file = str(row.get("source_file", "")).lower()

    # Strongest domestic leagues.
    if any(
        token in league
        for token in [
            "premier league",
            "la liga",
            "serie a",
            "bundesliga",
            "ligue 1",
        ]
    ):
        return 1.00

    # Strong European second tier of data relevance.
    if any(token in league for token in ["eredivisie", "liga portugal", "super lig"]):
        return 0.88

    # Explicit extra FBref sources without xG.
    if "fbref_saudi" in source_file:
        return 0.62

    if "fbref_liga_mx" in source_file:
        return 0.68

    if "fbref_mls" in source_file:
        return 0.66

    if "fbref_belgian" in source_file:
        return 0.76

    # Top-5 FBref file sometimes has league values like "eng Premier League".
    if "fbref_top5" in source_file:
        return 1.00

    # Kaggle extra includes mixed leagues. If league is missing/unknown, be moderate.
    if "kaggle_extra" in source_file:
        return 0.88

    return 0.85

def build_player_scoring_features(
    input_path: Path | None = None,
    output_path: Path | None = None,
    min_minutes: int = 0,
) -> pd.DataFrame:
    """
    Build player-level scoring features for Golden Boot modeling.

    Input:
    - data/processed/club_player_stats_normalized.csv

    Output:
    - data/processed/player_scoring_features.csv
    """
    input_path = input_path or DATA_PROCESSED / "club_player_stats_with_squads.csv"
    output_path = output_path or DATA_PROCESSED / "player_scoring_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run python -m src.data.normalize_club_player_sources first."
        )

    raw = pd.read_csv(input_path)

    required_columns = {
        "player",
        "team",
        "nation_raw",
        "position",
        "club",
        "league",
        "age",
        "matches_played",
        "starts",
        "minutes",
        "nineties",
        "goals",
        "assists",
        "non_penalty_goals",
        "penalties_scored",
        "penalties_attempted",
        "xg",
        "npxg",
        "xag",
        "shots",
    }

    missing = required_columns - set(raw.columns)
    if missing:
        raise ValueError(f"Missing expected normalized player columns: {missing}")

    df = raw.copy()

    numeric_columns = [
        "age",
        "matches_played",
        "starts",
        "minutes",
        "nineties",
        "goals",
        "assists",
        "non_penalty_goals",
        "penalties_scored",
        "penalties_attempted",
        "xg",
        "npxg",
        "xag",
        "shots",
    ]

    for col in numeric_columns:
        df[col] = _safe_numeric(df, col)

    df = df[df["minutes"] >= min_minutes].copy()

    df["nineties_safe"] = df["nineties"].where(df["nineties"] > 0, other=pd.NA)

    df["goals_per90"] = (df["goals"] / df["nineties_safe"]).fillna(0.0)
    df["non_penalty_goals_per90"] = (
        df["non_penalty_goals"] / df["nineties_safe"]
    ).fillna(0.0)
    df["xg_per90"] = (df["xg"] / df["nineties_safe"]).fillna(0.0)
    df["npxg_per90"] = (df["npxg"] / df["nineties_safe"]).fillna(0.0)
    df["shots_per90"] = (df["shots"] / df["nineties_safe"]).fillna(0.0)
    df["starts_per_match"] = (
        df["starts"] / df["matches_played"].where(df["matches_played"] > 0, other=pd.NA)
    ).fillna(0.0)

    df["is_penalty_taker_proxy"] = (df["penalties_attempted"] > 0).astype(int)

    # Some added FBref extra-league files only contain standard + shooting stats,
    # not expected-goals stats. In those files, xg/npxg/xag are unavailable, not
    # truly zero. For those rows, use a goals + shots fallback.
    source_file = df.get("source_file", "").astype(str).str.lower()

    xg_unavailable_sources = source_file.str.contains(
        "fbref_mls|fbref_liga_mx|fbref_saudi|fbref_belgian",
        case=False,
        na=False,
    )

    df["has_xg_data"] = ~xg_unavailable_sources

    df["scoring_signal_type"] = "xg_goals_shots"
    df.loc[~df["has_xg_data"], "scoring_signal_type"] = "goals_shots_fallback"

    df["club_scoring_score"] = 0.0

    # Normal case: xG is available.
    df.loc[df["has_xg_data"], "club_scoring_score"] = (
        0.45 * df.loc[df["has_xg_data"], "npxg_per90"]
        + 0.25 * df.loc[df["has_xg_data"], "xg_per90"]
        + 0.20 * df.loc[df["has_xg_data"], "non_penalty_goals_per90"]
        + 0.10 * df.loc[df["has_xg_data"], "shots_per90"].clip(upper=5.0) / 5.0
    )

    # Fallback case: xG is unavailable, so use non-penalty goals + shots.
    # The shots term is deliberately small: it helps distinguish high-volume
    # attackers but should not dominate actual scoring.
    df.loc[~df["has_xg_data"], "club_scoring_score"] = (
        0.70 * df.loc[~df["has_xg_data"], "non_penalty_goals_per90"]
        + 0.20 * df.loc[~df["has_xg_data"], "goals_per90"]
        + 0.10 * df.loc[~df["has_xg_data"], "shots_per90"].clip(upper=5.0) / 5.0
    )

    df["league_strength_multiplier"] = df.apply(_league_strength_multiplier, axis=1)

    df["club_scoring_score_raw"] = df["club_scoring_score"]

    df["club_scoring_score"] = (
        df["club_scoring_score"] * df["league_strength_multiplier"]
    )

    df["minutes_reliability"] = (df["minutes"] / 900.0).clip(upper=1.0)

    df["club_scoring_score_shrunk"] = (
        df["minutes_reliability"] * df["club_scoring_score"]
        + (1.0 - df["minutes_reliability"]) * 0.10
    )

    output_columns = [
        "player",
        "team",
        "nation_raw",
        "in_world_cup_squad",
        "squad_player",
        "squad_team",
        "position",
        "club",
        "league",
        "age",
        "matches_played",
        "starts",
        "minutes",
        "nineties",
        "goals",
        "assists",
        "non_penalty_goals",
        "penalties_scored",
        "penalties_attempted",
        "xg",
        "npxg",
        "xag",
        "shots",
        "goals_per90",
        "non_penalty_goals_per90",
        "xg_per90",
        "shots_per90",
        "has_xg_data",
        "scoring_signal_type",
        "starts_per_match",
        "starts_per_match",
        "is_penalty_taker_proxy",
        "club_scoring_score",
        "league_strength_multiplier",
        "club_scoring_score_raw",   
        "minutes_reliability",
        "club_scoring_score_shrunk",
        "source_file",
    ]

    output_columns = [col for col in output_columns if col in df.columns]

    out = df[output_columns].sort_values(
        by=["club_scoring_score_shrunk", "minutes"],
        ascending=[False, False],
    )

    out.to_csv(output_path, index=False)

    return out


if __name__ == "__main__":
    features = build_player_scoring_features()
    display_features = features.copy()

    if "in_world_cup_squad" in display_features.columns:
        display_features = display_features[display_features["in_world_cup_squad"] == 1]

    print(display_features.head(30).to_string(index=False))
    print(f"\nSaved {len(features):,} player rows to {DATA_PROCESSED / 'player_scoring_features.csv'}")