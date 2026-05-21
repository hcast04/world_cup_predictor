from pathlib import Path

import pandas as pd


def _format_probability(value: float) -> str:
    return f"{100 * value:.1f}%"


def dataframe_to_markdown_table(
    df: pd.DataFrame,
    probability_columns: list[str] | None = None,
    float_columns: list[str] | None = None,
) -> str:
    """
    Convert a DataFrame to a readable Markdown table.

    Probability columns are formatted as percentages.
    Float columns are formatted to two decimals.
    """
    probability_columns = probability_columns or []
    float_columns = float_columns or []

    display_df = df.copy()

    for col in probability_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_format_probability)

    for col in float_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}")

    return display_df.to_markdown(index=False)


def write_stage1_summary_report(
    winner_predictions: pd.DataFrame,
    golden_boot_predictions: pd.DataFrame,
    group_predictions: pd.DataFrame | None,
    output_path: Path,
    n_top: int = 15,
) -> None:
    """
    Write a compact Stage 1 Markdown report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append("# Stage 1 Prediction Summary")
    lines.append("")
    lines.append("This report summarizes the current Stage 1 prediction outputs.")
    lines.append("")
    lines.append("> Important: at this development stage, ratings and player inputs may still contain placeholders.")
    lines.append("")

    lines.append("## World Cup winner probabilities")
    lines.append("")
    winner_cols = [
        "team",
        "round_of_32_prob",
        "round_of_16_prob",
        "quarter_final_prob",
        "semi_final_prob",
        "final_prob",
        "winner_prob",
    ]
    winner_table = winner_predictions[winner_cols].head(n_top)
    lines.append(
        dataframe_to_markdown_table(
            winner_table,
            probability_columns=[
                "round_of_32_prob",
                "round_of_16_prob",
                "quarter_final_prob",
                "semi_final_prob",
                "final_prob",
                "winner_prob",
            ],
        )
    )
    lines.append("")

    lines.append("## Golden Boot probabilities")
    lines.append("")
    golden_cols = [
        "player",
        "team",
        "expected_goals_per_match",
        "expected_simulated_goals",
        "golden_boot_prob",
        "top_3_scorer_prob",
    ]
    golden_table = golden_boot_predictions[golden_cols].head(n_top)
    lines.append(
        dataframe_to_markdown_table(
            golden_table,
            probability_columns=["golden_boot_prob", "top_3_scorer_prob"],
            float_columns=["expected_goals_per_match", "expected_simulated_goals"],
        )
    )
    lines.append("")

    if group_predictions is not None and not group_predictions.empty:
        lines.append("## Group-stage qualification probabilities")
        lines.append("")
        group_cols = [
            col
            for col in [
                "team",
                "expected_points",
                "finish_1_prob",
                "finish_2_prob",
                "finish_3_prob",
                "finish_4_prob",
                "top_two_prob",
                "best_third_prob",
                "qualify_prob",
            ]
            if col in group_predictions.columns
        ]
        group_table = group_predictions[group_cols].head(n_top)
        prob_cols = [col for col in group_cols if col.endswith("_prob")]
        lines.append(
            dataframe_to_markdown_table(
                group_table,
                probability_columns=prob_cols,
                float_columns=["expected_points"],
            )
        )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_knockout_summary_report(
    predictions: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """
    Write a compact knockout-stage Markdown report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("This report summarizes manually entered knockout fixture predictions.")
    lines.append("")

    if predictions.empty:
        lines.append("No valid knockout predictions were available.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    cols = [
        "match_id",
        "stage",
        "team_a",
        "team_b",
        "team_a_advance_prob",
        "team_b_advance_prob",
        "team_a_win_90_prob",
        "draw_90_prob",
        "team_b_win_90_prob",
    ]
    cols = [col for col in cols if col in predictions.columns]

    lines.append(
        dataframe_to_markdown_table(
            predictions[cols],
            probability_columns=[
                "team_a_advance_prob",
                "team_b_advance_prob",
                "team_a_win_90_prob",
                "draw_90_prob",
                "team_b_win_90_prob",
            ],
        )
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")