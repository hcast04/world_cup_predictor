import pandas as pd

from src.models.elo_poisson import expected_goals_from_elo
from src.models.team_strength_baseline import predict_expected_goals_strength_baseline


class MatchEngine:
    """
    Unified expected-goals interface for different match models.

    Supported model_type values:
    - elo_poisson
    - strength_baseline
    """

    def __init__(
        self,
        model_type: str,
        elo_lookup: dict[str, float] | None = None,
        host_lookup: dict[str, int] | None = None,
        team_strengths: pd.DataFrame | None = None,
    ) -> None:
        self.model_type = model_type
        self.elo_lookup = elo_lookup or {}
        self.host_lookup = host_lookup or {}
        self.team_strengths = team_strengths

        supported = {"elo_poisson", "strength_baseline"}
        if model_type not in supported:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose one of {supported}")

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

    def expected_goals(self, team_a: str, team_b: str) -> tuple[float, float]:
        if self.model_type == "elo_poisson":
            return expected_goals_from_elo(
                elo_a=self.elo_lookup[team_a],
                elo_b=self.elo_lookup[team_b],
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

        Elo is natural for the Elo model. For the strength model, use attack quality
        and defensive quality as a crude combined score.
        """
        if self.model_type == "elo_poisson":
            return float(self.elo_lookup.get(team, 1500.0))

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