import pandas as pd

from src.data.loaders import PROJECT_ROOT


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    input_path = PROJECT_ROOT / "outputs" / "tables" / "stage1_model_comparison_winner_probs.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run python scripts/compare_stage1_models.py first."
        )

    df = pd.read_csv(input_path)

    output_path = PROJECT_ROOT / "outputs" / "tables" / "stage1_model_comparison_report.md"

    lines = []
    lines.append("# Stage 1 Model Comparison")
    lines.append("")
    lines.append("This report compares Stage 1 winner probabilities from two model variants:")
    lines.append("")
    lines.append("- `elo_poisson`: uses current Elo-style ratings.")
    lines.append("- `strength_baseline`: uses historical attack/defence strengths estimated from international results.")
    lines.append("")
    lines.append("> The comparison is only as reliable as the current input data. Placeholder ratings and incomplete player data may still affect results.")
    lines.append("")

    display_cols = [
        "team",
        "winner_prob_elo",
        "winner_prob_strength",
        "winner_prob_diff_strength_minus_elo",
    ]

    higher_strength = df.sort_values(
        "winner_prob_diff_strength_minus_elo",
        ascending=False,
    ).head(15)[display_cols].copy()

    higher_elo = df.sort_values(
        "winner_prob_diff_strength_minus_elo",
        ascending=True,
    ).head(15)[display_cols].copy()

    for col in display_cols:
        if col != "team":
            higher_strength[col] = higher_strength[col].map(pct)
            higher_elo[col] = higher_elo[col].map(pct)

    lines.append("## Teams rated higher by `strength_baseline`")
    lines.append("")
    lines.append(higher_strength.to_markdown(index=False))
    lines.append("")

    lines.append("## Teams rated higher by `elo_poisson`")
    lines.append("")
    lines.append(higher_elo.to_markdown(index=False))
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved model comparison report to:")
    print(output_path)


if __name__ == "__main__":
    main()