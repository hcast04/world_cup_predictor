from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW, DATA_PROCESSED


NATION_TO_TEAM = {
    "ar ARG": "Argentina",
    "au AUS": "Australia",
    "at AUT": "Austria",
    "be BEL": "Belgium",
    "ba BIH": "Bosnia and Herzegovina",
    "br BRA": "Brazil",
    "cm CAN": "Canada",
    "ca CAN": "Canada",
    "cv CPV": "Cape Verde",
    "co COL": "Colombia",
    "hr CRO": "Croatia",
    "cw CUW": "Curacao",
    "cz CZE": "Czechia",
    "cd COD": "DR Congo",
    "ec ECU": "Ecuador",
    "eg EGY": "Egypt",
    "eng ENG": "England",
    "fr FRA": "France",
    "de GER": "Germany",
    "gh GHA": "Ghana",
    "ht HAI": "Haiti",
    "ir IRN": "Iran",
    "iq IRQ": "Iraq",
    "ci CIV": "Ivory Coast",
    "jp JPN": "Japan",
    "jo JOR": "Jordan",
    "kr KOR": "South Korea",
    "mx MEX": "Mexico",
    "ma MAR": "Morocco",
    "nl NED": "Netherlands",
    "nz NZL": "New Zealand",
    "no NOR": "Norway",
    "pa PAN": "Panama",
    "py PAR": "Paraguay",
    "pt POR": "Portugal",
    "qa QAT": "Qatar",
    "sa KSA": "Saudi Arabia",
    "sct SCO": "Scotland",
    "sn SEN": "Senegal",
    "za RSA": "South Africa",
    "es ESP": "Spain",
    "se SWE": "Sweden",
    "ch SUI": "Switzerland",
    "tn TUN": "Tunisia",
    "tr TUR": "Turkey",
    "us USA": "United States",
    "uy URU": "Uruguay",
    "uz UZB": "Uzbekistan",
    "dz ALG": "Algeria",
}


def _clean_nation_to_team(nation: str) -> str:
    """
    Convert FBref/Kaggle nation strings into project team names.

    Examples:
    - 'us USA' -> 'United States'
    - 'ma MAR' -> 'Morocco'
    """
    nation = str(nation).strip()

    if nation in NATION_TO_TEAM:
        return NATION_TO_TEAM[nation]

    # Fallback: use the final token if unknown.
    return nation


def _safe_numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index)

    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def build_player_scoring_features(
    input_path: Path | None = None,
    output_path: Path | None = None,
    min_minutes: int = 0,
) -> pd.DataFrame:
    """
    Build player-level scoring features for Golden Boot modeling.

    Input:
    - data/raw/club_player_stats_2025_2026.csv

    Output:
    - data/processed/player_scoring_features.csv
    """
    input_path = input_path or DATA_RAW / "club_player_stats_2025_2026.csv"
    output_path = output_path or DATA_PROCESSED / "player_scoring_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Save the club player stats CSV there first."
        )

    raw = pd.read_csv(input_path)

    required_columns = {
        "Player",
        "Nation",
        "Pos",
        "Squad",
        "Comp",
        "Age",
        "MP",
        "Starts",
        "Min",
        "90s",
        "Gls",
        "Ast",
        "G-PK",
        "PK",
        "PKatt",
        "xG",
        "npxG",
        "xAG",
        "Sh",
    }

    missing = required_columns - set(raw.columns)
    if missing:
        raise ValueError(f"Missing expected club player columns: {missing}")

    df = pd.DataFrame()

    df["player"] = raw["Player"].astype(str).str.strip()
    df["nation_raw"] = raw["Nation"].astype(str).str.strip()
    df["team"] = df["nation_raw"].map(_clean_nation_to_team)
    df["position"] = raw["Pos"].astype(str).str.strip()
    df["club"] = raw["Squad"].astype(str).str.strip()
    df["league"] = raw["Comp"].astype(str).str.strip()

    df["age"] = _safe_numeric(raw, "Age")
    df["matches_played"] = _safe_numeric(raw, "MP")
    df["starts"] = _safe_numeric(raw, "Starts")
    df["minutes"] = _safe_numeric(raw, "Min")
    df["nineties"] = _safe_numeric(raw, "90s")

    df["goals"] = _safe_numeric(raw, "Gls")
    df["assists"] = _safe_numeric(raw, "Ast")
    df["non_penalty_goals"] = _safe_numeric(raw, "G-PK")
    df["penalties_scored"] = _safe_numeric(raw, "PK")
    df["penalties_attempted"] = _safe_numeric(raw, "PKatt")
    df["xg"] = _safe_numeric(raw, "xG")
    df["npxg"] = _safe_numeric(raw, "npxG")
    df["xag"] = _safe_numeric(raw, "xAG")
    df["shots"] = _safe_numeric(raw, "Sh")

    df = df[df["minutes"] >= min_minutes].copy()

    # Avoid divide-by-zero for very low-minute players.
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

    # Main scoring signal for Golden Boot.
    # npxG is preferred because penalties are handled separately.
    df["club_scoring_score"] = (
        0.45 * df["npxg_per90"]
        + 0.25 * df["xg_per90"]
        + 0.20 * df["non_penalty_goals_per90"]
        + 0.10 * df["shots_per90"].clip(upper=5.0) / 5.0
    )

    # Minute reliability matters. Someone with 40 minutes and a goal should not dominate.
    df["minutes_reliability"] = (df["minutes"] / 900.0).clip(upper=1.0)

    df["club_scoring_score_shrunk"] = (
        df["minutes_reliability"] * df["club_scoring_score"]
        + (1.0 - df["minutes_reliability"]) * 0.10
    )

    output_columns = [
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
        "goals_per90",
        "non_penalty_goals_per90",
        "xg_per90",
        "npxg_per90",
        "shots_per90",
        "starts_per_match",
        "is_penalty_taker_proxy",
        "club_scoring_score",
        "minutes_reliability",
        "club_scoring_score_shrunk",
    ]

    out = df[output_columns].sort_values(
        by=["club_scoring_score_shrunk", "minutes"],
        ascending=[False, False],
    )

    out.to_csv(output_path, index=False)

    return out


if __name__ == "__main__":
    features = build_player_scoring_features()
    print(features.head(30).to_string(index=False))
    print(f"\nSaved {len(features):,} player rows to {DATA_PROCESSED / 'player_scoring_features.csv'}")