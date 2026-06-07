from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW, DATA_PROCESSED


SOURCE_DIR = DATA_RAW / "club_player_stats_sources"


NATION_TO_TEAM = {
    "ar ARG": "Argentina",
    "au AUS": "Australia",
    "at AUT": "Austria",
    "be BEL": "Belgium",
    "ba BIH": "Bosnia and Herzegovina",
    "br BRA": "Brazil",
    "ca CAN": "Canada",
    "cm CAN": "Canada",
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
        "it ITA": "Italy",
    "dk DEN": "Denmark",
    "ng NGA": "Nigeria",
    "rs SRB": "Serbia",
    "pl POL": "Poland",
    "ml MLI": "Mali",
    "cm CMR": "Cameroon",
    "xk KVX": "Kosovo",
    "gr GRE": "Greece",
    "al ALB": "Albania",
    "ie IRL": "Republic of Ireland",
    "ua UKR": "Ukraine",
    "wls WAL": "Wales",
    "ao ANG": "Angola",
    "ge GEO": "Georgia",
    "is ISL": "Iceland",
    "ro ROU": "Romania",
    "hu HUN": "Hungary",
    "sk SVK": "Slovakia",
    "bf BFA": "Burkina Faso",
    "gn GUI": "Guinea",
    "me MNE": "Montenegro",
    "ve VEN": "Venezuela",
    "cl CHI": "Chile",
    "tg TOG": "Togo",
    "ga GAB": "Gabon",
    "si SVN": "Slovenia",
    "nir NIR": "Northern Ireland",
    "lu LUX": "Luxembourg",
    "gm GAM": "Gambia",
    "do DOM": "Dominican Republic",
    "gw GNB": "Guinea-Bissau",
    "id IDN": "Indonesia",
    "ly LBY": "Libya",
    "ru RUS": "Russia",
    "gp GLP": "Guadeloupe",
    "gq EQG": "Equatorial Guinea",
    "zw ZIM": "Zimbabwe",
    "mk MKD": "North Macedonia",
    "bj BEN": "Benin",
    "mr MTN": "Mauritania",
    "am ARM": "Armenia",
    "cf CTA": "Central African Republic",
}


COLUMN_ALIASES = {
    "player": ["Player", "player", "name", "Name", "shortName"],
    "nation_raw": ["Nation", "nation", "nationality", "Nationality", "country"],
    "position": ["Pos", "position", "Position"],
    "club": ["Squad", "squad", "club", "Club", "team", "Team"],
    "league": ["Comp", "competition", "league", "League", "tournament"],
    "age": ["Age", "age"],
    "matches_played": ["MP", "appearances", "matches", "matches_played"],
    "starts": ["Starts", "starts"],
    "minutes": ["Min", "minutes", "Minutes"],
    "nineties": ["90s", "nineties"],
    "goals": ["Gls", "goals", "Goals"],
    "assists": ["Ast", "assists", "Assists"],
    "non_penalty_goals": ["G-PK", "non_penalty_goals", "npg", "np_goals"],
    "penalties_scored": ["PK", "penalties_scored", "penalty_goals"],
    "penalties_attempted": ["PKatt", "penalties_attempted"],
    "xg": ["xG", "xg", "expected_goals"],
    "npxg": ["npxG", "npxg", "non_penalty_xg"],
    "xag": ["xAG", "xag", "expected_assists", "xA"],
    "shots": ["Sh", "shots", "Shots"],
}


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for col in aliases:
        if col in df.columns:
            return col
    return None


def _get_series(
    df: pd.DataFrame,
    canonical_col: str,
    default=None,
) -> pd.Series:
    source_col = _find_column(df, COLUMN_ALIASES[canonical_col])

    if source_col is None:
        return pd.Series(default, index=df.index)

    return df[source_col]


def _to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _nation_to_team(value: str) -> str:
    value = str(value).strip()

    if value in NATION_TO_TEAM:
        return NATION_TO_TEAM[value]

    # FBref format often ends with the 3-letter code. Keep raw if unknown.
    return value


def normalize_one_player_source(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)

    out = pd.DataFrame()
    out["player"] = _get_series(raw, "player", "").astype(str).str.strip()
    out["nation_raw"] = _get_series(raw, "nation_raw", "").astype(str).str.strip()
    out["team"] = out["nation_raw"].map(_nation_to_team)

    out["position"] = _get_series(raw, "position", "").astype(str).str.strip()
    out["club"] = _get_series(raw, "club", "").astype(str).str.strip()
    out["league"] = _get_series(raw, "league", "").astype(str).str.strip()

    out["age"] = _to_numeric(_get_series(raw, "age", 0))
    out["matches_played"] = _to_numeric(_get_series(raw, "matches_played", 0))
    out["starts"] = _to_numeric(_get_series(raw, "starts", 0))
    out["minutes"] = _to_numeric(_get_series(raw, "minutes", 0))
    out["nineties"] = _to_numeric(_get_series(raw, "nineties", 0))

    out["goals"] = _to_numeric(_get_series(raw, "goals", 0))
    out["assists"] = _to_numeric(_get_series(raw, "assists", 0))
    out["non_penalty_goals"] = _to_numeric(
        _get_series(raw, "non_penalty_goals", out["goals"])
    )
    out["penalties_scored"] = _to_numeric(_get_series(raw, "penalties_scored", 0))
    out["penalties_attempted"] = _to_numeric(_get_series(raw, "penalties_attempted", 0))
    out["xg"] = _to_numeric(_get_series(raw, "xg", 0))
    out["npxg"] = _to_numeric(_get_series(raw, "npxg", out["xg"]))
    out["xag"] = _to_numeric(_get_series(raw, "xag", 0))
    out["shots"] = _to_numeric(_get_series(raw, "shots", 0))

    # If 90s is missing, derive from minutes.
    out["nineties"] = out["nineties"].where(
        out["nineties"] > 0,
        out["minutes"] / 90.0,
    )

    out["source_file"] = path.name

    # Drop empty player rows.
    out = out[out["player"].astype(str).str.len() > 0].copy()

    return out


def normalize_all_player_sources(
    source_dir: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    source_dir = source_dir or SOURCE_DIR
    output_path = output_path or DATA_PROCESSED / "club_player_stats_normalized.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    files = sorted(
        path
        for path in source_dir.glob("*.csv")
        if path.name not in {
            "kaggle_extra_profiles.csv",
            "kaggle_extra_stats.csv",
        }
    )

    if not files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    frames = []

    for path in files:
        print(f"Normalizing {path.name}...")
        frame = normalize_one_player_source(path)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    combined = combined.sort_values(
        by=["player", "team", "minutes"],
        ascending=[True, True, False],
    )

    # If same player/team appears in multiple sources, keep the row with most minutes.
    combined = combined.drop_duplicates(
        subset=["player", "team"],
        keep="first",
    ).reset_index(drop=True)

    combined.to_csv(output_path, index=False)

    return combined


if __name__ == "__main__":
    normalized = normalize_all_player_sources()
    print(normalized.head(30).to_string(index=False))
    print(f"\nSaved {len(normalized):,} rows to {DATA_PROCESSED / 'club_player_stats_normalized.csv'}")