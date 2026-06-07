import pandas as pd

from src.data.loaders import DATA_PROCESSED
from src.models.elo_poisson import expected_goals_from_elo
from src.models.team_strength_baseline import predict_expected_goals_strength_baseline


def load_squad_elo_adjustments() -> dict[str, float]:
    """
    Load squad-based Elo residual adjustments.

    If the file does not exist, return an empty dictionary so the Elo model
    behaves exactly like the original raw-Elo version.
    """
    path = DATA_PROCESSED / "team_squad_strength.csv"

    if not path.exists():
        return {}

    df = pd.read_csv(path)

    required = {"team", "squad_elo_adjustment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in team_squad_strength.csv: {missing}")

    return dict(
        zip(
            df["team"].astype(str),
            pd.to_numeric(df["squad_elo_adjustment"], errors="coerce").fillna(0.0),
        )
    )


class MatchEngine:
    """
    Unified expected-goals interface for different match models.

    Supported model_type values:
    - elo_poisson
    - strength_baseline

    For elo_poisson, the model uses:
        effective_elo = base_elo + squad_elo_adjustment

    if data/processed/team_squad_strength.csv exists.
    """

    def __init__(
        self,
        model_type: str,
        elo_lookup: dict[str, float] | None = None,
        host_lookup: dict[str, int] | None = None,
        team_strengths: pd.DataFrame | None = None,
        squad_elo_adjustments: dict[str, float] | None = None,
    ) -> None:
        self.model_type = model_type
        self.elo_lookup = elo_lookup or {}
        self.host_lookup = host_lookup or {}
        self.team_strengths = team_strengths

        if squad_elo_adjustments is None:
            self.squad_elo_adjustments = load_squad_elo_adjustments()
        else:
            self.squad_elo_adjustments = squad_elo_adjustments

        supported = {"elo_poisson", "strength_baseline"}
        if model_type not in supported:
            raise ValueError(
                f"Unsupported model_type: {model_type}. Choose one of {supported}"
            )

        if model_type == "elo_poisson" and not self.elo_lookup:
            raise ValueError("elo_lookup is required for elo_poisson model.")

        if model_type == "strength_baseline" and team_strengths is None:
            raise ValueError("team_strengths is required for strength_baseline model.")

    def has_team(self, team: str) -> bool:
        if self.model_type == "elo_poisson":
            return team in self.elo_lookup

        if self.model_type == "strength_baseline":
            assert self.team_strengths is not None
            return team in set(self.team_strengths["team"])

        return False

    def effective_elo(self, team: str) -> float:
        """
        Return base Elo plus squad adjustment.

        If the team has no squad adjustment, use raw Elo.
        """
        base_elo = float(self.elo_lookup.get(team, 1500.0))
        squad_adjustment = float(self.squad_elo_adjustments.get(team, 0.0))
        return base_elo + squad_adjustment

    def expected_goals(self, team_a: str, team_b: str) -> tuple[float, float]:
        if self.model_type == "elo_poisson":
            elo_a = self.effective_elo(team_a)
            elo_b = self.effective_elo(team_b)

            return expected_goals_from_elo(
                elo_a=elo_a,
                elo_b=elo_b,
                team_a_is_host=bool(self.host_lookup.get(team_a, 0)),
                team_b_is_host=bool(self.host_lookup.get(team_b, 0)),
            )

        if self.model_type == "strength_baseline":
            return predict_expected_goals_strength_baseline(
                team_a=team_a,
                team_b=team_b,
                strengths=self.team_strengths,
            )

        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def rating_for_tiebreak(self, team: str) -> float:
        """
        Return a numeric team strength for generic knockout seeding/tiebreaks.

        For the Elo model, use effective Elo, not raw Elo.
        """
        if self.model_type == "elo_poisson":
            return self.effective_elo(team)

        if self.model_type == "strength_baseline":
            assert self.team_strengths is not None
            strengths = self.team_strengths.set_index("team")

            if team not in strengths.index:
                return 1000.0

            row = strengths.loc[team]
            attack = float(row["attack_index"])
            defence = float(row["defence_index"])

            # Higher attack is better. Lower defence_index is better.
            return 1000.0 * attack / max(defence, 0.25)

        return 1000.0