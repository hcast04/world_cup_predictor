import pandas as pd

from src.data.loaders import PROJECT_ROOT


def main() -> None:
    predictions_dir = PROJECT_ROOT / "outputs" / "predictions"

    baseline_path = predictions_dir / "group_stage_qualification_probabilities.csv"
    recent_form_path = predictions_dir / "recent_form_group_stage_probabilities.csv"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing {baseline_path}")

    if not recent_form_path.exists():
        raise FileNotFoundError(f"Missing {recent_form_path}")

    baseline = pd.read_csv(baseline_path)
    recent = pd.read_csv(recent_form_path)

    keep_cols = ["team", "expected_points", "top_two_prob", "best_third_prob", "qualify_prob"]

    comparison = baseline[keep_cols].merge(
        recent[keep_cols],
        on="team",
        suffixes=("_baseline", "_recent_form"),
    )

    comparison["qualify_prob_diff_recent_minus_baseline"] = (
        comparison["qualify_prob_recent_form"] - comparison["qualify_prob_baseline"]
    )

    comparison["expected_points_diff_recent_minus_baseline"] = (
        comparison["expected_points_recent_form"] - comparison["expected_points_baseline"]
    )

    comparison = comparison.sort_values(
        "qualify_prob_diff_recent_minus_baseline",
        ascending=False,
    )

    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "tables"
        / "group_stage_model_comparison.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    print("\nGroup-stage model comparison")
    print("----------------------------")
    print("\nTeams helped most by recent-form model:")
    print(
        comparison[
            [
                "team",
                "qualify_prob_baseline",
                "qualify_prob_recent_form",
                "qualify_prob_diff_recent_minus_baseline",
            ]
        ].head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\nTeams hurt most by recent-form model:")
    print(
        comparison[
            [
                "team",
                "qualify_prob_baseline",
                "qualify_prob_recent_form",
                "qualify_prob_diff_recent_minus_baseline",
            ]
        ].tail(20).sort_values("qualify_prob_diff_recent_minus_baseline")
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print(f"\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()