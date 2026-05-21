from dataclasses import dataclass

import pandas as pd

from src.simulation.match import MatchResult


@dataclass(frozen=True)
class TeamStanding:
    team: str
    played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    goal_difference: int
    points: int


def initialize_group_table(teams: list[str]) -> dict[str, dict[str, int]]:
    """
    Create an empty group table for a list of teams.
    """
    return {
        team: {
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "points": 0,
        }
        for team in teams
    }


def update_group_table(
    table: dict[str, dict[str, int]],
    result: MatchResult,
) -> dict[str, dict[str, int]]:
    """
    Update a group table with one simulated match result.
    """
    team_a = result.team_a
    team_b = result.team_b
    goals_a = result.goals_a
    goals_b = result.goals_b

    table[team_a]["played"] += 1
    table[team_b]["played"] += 1

    table[team_a]["goals_for"] += goals_a
    table[team_a]["goals_against"] += goals_b

    table[team_b]["goals_for"] += goals_b
    table[team_b]["goals_against"] += goals_a

    if goals_a > goals_b:
        table[team_a]["wins"] += 1
        table[team_b]["losses"] += 1
        table[team_a]["points"] += 3
    elif goals_b > goals_a:
        table[team_b]["wins"] += 1
        table[team_a]["losses"] += 1
        table[team_b]["points"] += 3
    else:
        table[team_a]["draws"] += 1
        table[team_b]["draws"] += 1
        table[team_a]["points"] += 1
        table[team_b]["points"] += 1

    table[team_a]["goal_difference"] = (
        table[team_a]["goals_for"] - table[team_a]["goals_against"]
    )
    table[team_b]["goal_difference"] = (
        table[team_b]["goals_for"] - table[team_b]["goals_against"]
    )

    return table


def table_to_dataframe(table: dict[str, dict[str, int]]) -> pd.DataFrame:
    """
    Convert a group table dictionary to a ranked DataFrame.

    Baseline tie-breakers:
    1. points
    2. goal difference
    3. goals for
    4. team name alphabetical, only to make the output deterministic

    Later we can replace the final alphabetical rule with FIFA's full tie-breaker logic.
    """
    rows = []

    for team, stats in table.items():
        row = {"team": team}
        row.update(stats)
        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["points", "goal_difference", "goals_for", "team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    df.insert(0, "position", range(1, len(df) + 1))

    return df


def build_group_table(
    teams: list[str],
    results: list[MatchResult],
) -> pd.DataFrame:
    """
    Build a ranked group table from a list of match results.
    """
    table = initialize_group_table(teams)

    for result in results:
        table = update_group_table(table, result)

    return table_to_dataframe(table)