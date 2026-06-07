from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW, DATA_PROCESSED


SOURCE_DIR = DATA_RAW / "club_player_stats_sources"


def normalize_kaggle_extra_player_stats(
    profiles_path: Path | None = None,
    stats_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Normalize the Kaggle extra player dataset with separate profile and stat files.

    Expected profile columns:
    - player_id
    - name
    - league
    - position
    - market_value

    Expected stat columns:
    - player_id
    - league
    - appearances
    - matches_started
    - minutes_played
    - goals
    - assists
    - expected_goals
    - expected_assists
    - rating
    - total_shots
    - shots_on_target
    - yellow_cards
    - red_cards
    - tackles
    - interceptions
    - saves

    Important:
    This source does not include nationality in the example schema.
    Therefore, team/nation_raw are left blank here and can be filled later by
    name matching or a squad/nationality source.
    """
    profiles_path = profiles_path or SOURCE_DIR / "kaggle_extra_profiles.csv"
    stats_path = stats_path or SOURCE_DIR / "kaggle_extra_stats.csv"
    output_path = output_path or SOURCE_DIR / "kaggle_extra_normalized.csv"

    if not profiles_path.exists():
        raise FileNotFoundError(f"Missing profiles file: {profiles_path}")

    if not stats_path.exists():
        raise FileNotFoundError(f"Missing stats file: {stats_path}")

    profiles = pd.read_csv(profiles_path)
    stats = pd.read_csv(stats_path)

    required_profile_columns = {
        "player_id",
        "name",
        "league",
        "position",
        "market_value",
    }

    required_stat_columns = {
        "player_id",
        "league",
        "appearances",
        "matches_started",
        "minutes_played",
        "goals",
        "assists",
        "expected_goals",
        "expected_assists",
        "rating",
        "total_shots",
        "shots_on_target",
    }

    missing_profiles = required_profile_columns - set(profiles.columns)
    missing_stats = required_stat_columns - set(stats.columns)

    if missing_profiles:
        raise ValueError(f"Missing profile columns: {missing_profiles}")

    if missing_stats:
        raise ValueError(f"Missing stat columns: {missing_stats}")

    merged = stats.merge(
        profiles,
        on=["player_id", "league"],
        how="left",
        validate="many_to_one",
    )

    out = pd.DataFrame()

    out["player"] = merged["name"].astype(str).str.strip()
    out["nation_raw"] = ""
    out["team"] = ""
    out["position"] = merged["position"].astype(str).str.strip()
    out["club"] = ""
    out["league"] = merged["league"].astype(str).str.strip()

    out["age"] = 0
    out["matches_played"] = pd.to_numeric(merged["appearances"], errors="coerce").fillna(0)
    out["starts"] = pd.to_numeric(merged["matches_started"], errors="coerce").fillna(0)
    out["minutes"] = pd.to_numeric(merged["minutes_played"], errors="coerce").fillna(0)
    out["nineties"] = out["minutes"] / 90.0

    out["goals"] = pd.to_numeric(merged["goals"], errors="coerce").fillna(0)
    out["assists"] = pd.to_numeric(merged["assists"], errors="coerce").fillna(0)

    # This source does not provide penalties or non-penalty xG in the example.
    # Use goals/xG as fallback. This is imperfect but better than dropping the source.
    out["non_penalty_goals"] = out["goals"]
    out["penalties_scored"] = 0
    out["penalties_attempted"] = 0

    out["xg"] = pd.to_numeric(merged["expected_goals"], errors="coerce").fillna(0)
    out["npxg"] = out["xg"]
    out["xag"] = pd.to_numeric(merged["expected_assists"], errors="coerce").fillna(0)
    out["shots"] = pd.to_numeric(merged["total_shots"], errors="coerce").fillna(0)

    out["market_value"] = pd.to_numeric(merged["market_value"], errors="coerce").fillna(0)
    out["rating"] = pd.to_numeric(merged["rating"], errors="coerce").fillna(0)
    out["source_file"] = "kaggle_extra_profiles_stats"

    # Drop invalid player names.
    out = out[out["player"].str.len() > 0].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    return out


if __name__ == "__main__":
    normalized = normalize_kaggle_extra_player_stats()
    print(normalized.head(30).to_string(index=False))
    print(f"\nSaved {len(normalized):,} rows to {SOURCE_DIR / 'kaggle_extra_normalized.csv'}")