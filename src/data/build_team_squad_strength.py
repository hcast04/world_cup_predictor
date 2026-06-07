from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_PROCESSED


def _safe_numeric(series: pd.Series | float | int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _position_tokens(position: str) -> set[str]:
    """
    Convert position strings into clean tokens.

    Important:
    - Do not use substring checks like "F" in "DF".
    - "DF" should be defence, not attack.
    """
    position = (
        str(position)
        .upper()
        .replace("/", ",")
        .replace("-", ",")
        .replace(" ", "")
        .strip()
    )

    return {token for token in position.split(",") if token}


def _position_group(position: str) -> str:
    tokens = _position_tokens(position)

    if tokens & {"FW", "F"}:
        return "attack"

    if tokens & {"MF", "M"}:
        return "midfield"

    if tokens & {"DF", "D"}:
        return "defence"

    if tokens & {"GK", "G"}:
        return "goalkeeper"

    return "unknown"


def build_team_squad_strength(
    player_features_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build conservative squad-level strength estimates from player features.

    The purpose is not to replace Elo. Elo remains the main team-strength signal.

    This file creates a residual squad adjustment:
        effective_elo = base_elo + squad_elo_adjustment

    Design choices:
    - Only World Cup squad players are used.
    - Attack is weighted most heavily because goals/xG/shots are available.
    - Midfield and defence are given low weight because current data mostly uses
      minutes/league proxies rather than true defensive or ball-progression quality.
    - Incomplete units are shrunk toward conservative priors.
    - Final adjustment is capped to avoid overpowering Elo.
    """
    player_features_path = (
        player_features_path
        or DATA_PROCESSED / "player_scoring_features.csv"
    )
    output_path = output_path or DATA_PROCESSED / "team_squad_strength.csv"

    if not player_features_path.exists():
        raise FileNotFoundError(
            f"Missing {player_features_path}. "
            "Run python -m src.data.build_player_scoring_features first."
        )

    df = pd.read_csv(player_features_path)

    required = {
        "team",
        "player",
        "position",
        "in_world_cup_squad",
        "minutes",
        "club_scoring_score_shrunk",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in player scoring features: {missing}")

    df = df[df["in_world_cup_squad"].eq(1)].copy()
    df = df[df["team"].notna()].copy()

    df["minutes"] = _safe_numeric(df["minutes"])
    df["club_scoring_score_shrunk"] = _safe_numeric(
        df["club_scoring_score_shrunk"]
    )

    if "goals_per90" in df.columns:
        df["goals_per90"] = _safe_numeric(df["goals_per90"])
    else:
        df["goals_per90"] = 0.0

    if "xg_per90" in df.columns:
        df["xg_per90"] = _safe_numeric(df["xg_per90"])
    else:
        df["xg_per90"] = 0.0

    if "league_strength_multiplier" in df.columns:
        df["league_strength_multiplier"] = _safe_numeric(
            df["league_strength_multiplier"]
        )
        df["league_strength_multiplier"] = df["league_strength_multiplier"].replace(
            0.0, 0.85
        )
    else:
        df["league_strength_multiplier"] = 0.85

    df["position_group"] = df["position"].map(_position_group)

    # Reliability from club minutes.
    # This prevents tiny-sample rows from dominating.
    df["minutes_reliability"] = (df["minutes"] / 1200.0).clip(
        lower=0.0,
        upper=1.0,
    )

    # Generic player quality proxy for midfield/defence.
    # This is deliberately rough, so it should not dominate the final score.
    df["player_quality_proxy"] = (
        0.60 * df["minutes_reliability"]
        + 0.40 * df["league_strength_multiplier"]
    )

    # -------------------------
    # Attack
    # -------------------------
    attack = df[df["position_group"].eq("attack")].copy()

    attack = attack.sort_values(
        by=["team", "club_scoring_score_shrunk", "minutes"],
        ascending=[True, False, False],
    )

    attack_top = attack.groupby("team").head(5).copy()

    attack_scores = (
        attack_top.groupby("team")
        .agg(
            squad_attack_score=("club_scoring_score_shrunk", "mean"),
            elite_attacker_score=("club_scoring_score_shrunk", "max"),
            attacking_depth=("player", "count"),
        )
        .reset_index()
    )

    # -------------------------
    # Midfield
    # -------------------------
    midfield = df[df["position_group"].eq("midfield")].copy()

    midfield = midfield.sort_values(
        by=["team", "player_quality_proxy", "minutes"],
        ascending=[True, False, False],
    )

    midfield_top = midfield.groupby("team").head(6).copy()

    midfield_scores = (
        midfield_top.groupby("team")
        .agg(
            squad_midfield_score=("player_quality_proxy", "mean"),
            midfield_depth=("player", "count"),
        )
        .reset_index()
    )

    # -------------------------
    # Defence + goalkeeper
    # -------------------------
    defence = df[df["position_group"].isin(["defence", "goalkeeper"])].copy()

    defence = defence.sort_values(
        by=["team", "player_quality_proxy", "minutes"],
        ascending=[True, False, False],
    )

    defence_top = defence.groupby("team").head(6).copy()

    defence_scores = (
        defence_top.groupby("team")
        .agg(
            squad_defence_score=("player_quality_proxy", "mean"),
            defence_depth=("player", "count"),
        )
        .reset_index()
    )

    teams = pd.DataFrame({"team": sorted(df["team"].dropna().unique())})

    out = (
        teams.merge(attack_scores, on="team", how="left")
        .merge(midfield_scores, on="team", how="left")
        .merge(defence_scores, on="team", how="left")
    )

    # Conservative priors for missing units.
    # These are intentionally below average so missing data does not look like strength.
    fill_values = {
        "squad_attack_score": 0.20,
        "elite_attacker_score": 0.20,
        "attacking_depth": 0,
        "squad_midfield_score": 0.60,
        "midfield_depth": 0,
        "squad_defence_score": 0.60,
        "defence_depth": 0,
    }
    out = out.fillna(fill_values)

    # Depth factors: incomplete units are shrunk toward conservative priors.
    out["attack_depth_factor"] = (out["attacking_depth"] / 5.0).clip(
        lower=0.0,
        upper=1.0,
    )
    out["midfield_depth_factor"] = (out["midfield_depth"] / 4.0).clip(
        lower=0.0,
        upper=1.0,
    )
    out["defence_depth_factor"] = (out["defence_depth"] / 4.0).clip(
        lower=0.0,
        upper=1.0,
    )

    out["squad_attack_score_adj"] = (
        out["attack_depth_factor"] * out["squad_attack_score"]
        + (1.0 - out["attack_depth_factor"]) * 0.18
    )

    out["elite_attacker_score_adj"] = (
        out["attack_depth_factor"] * out["elite_attacker_score"]
        + (1.0 - out["attack_depth_factor"]) * 0.18
    )

    out["squad_midfield_score_adj"] = (
        out["midfield_depth_factor"] * out["squad_midfield_score"]
        + (1.0 - out["midfield_depth_factor"]) * 0.45
    )

    out["squad_defence_score_adj"] = (
        out["defence_depth_factor"] * out["squad_defence_score"]
        + (1.0 - out["defence_depth_factor"]) * 0.45
    )

    # Attack-heavy composite.
    # Midfield/defence are low-weight because they are currently noisy proxies.
    out["squad_strength_score_raw"] = (
        0.65 * out["squad_attack_score_adj"]
        + 0.25 * out["elite_attacker_score_adj"]
        + 0.05 * out["squad_midfield_score_adj"]
        + 0.05 * out["squad_defence_score_adj"]
    )

    # Center around the tournament average.
    mean_score = out["squad_strength_score_raw"].mean()
    std_score = out["squad_strength_score_raw"].std(ddof=0)

    if std_score == 0:
        out["squad_strength_z"] = 0.0
    else:
        out["squad_strength_z"] = (
            out["squad_strength_score_raw"] - mean_score
        ) / std_score

    # Conservative residual Elo adjustment.
    # Elo already captures most team strength; this should only fine-tune it.
    out["squad_elo_adjustment"] = (out["squad_strength_z"] * 18.0).clip(
        lower=-45.0,
        upper=45.0,
    )

    out = out.sort_values("squad_elo_adjustment", ascending=False).reset_index(
        drop=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Saved team squad strength to {output_path}")
    print(out.to_string(index=False))

    return out


if __name__ == "__main__":
    build_team_squad_strength()