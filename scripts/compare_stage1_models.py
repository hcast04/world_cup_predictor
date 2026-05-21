import pandas as pd

from src.data.loaders import PROJECT_ROOT


def load_predictions(model_type: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "predictions" / f"stage1_winner_predictions_{model_type}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run Stage 1 first with --model-type {model_type}."
        )

    df = pd.read_csv(path)
    keep_cols = [
        "team",
        "round_of_32_prob",
        "round_of_16_prob",
        "quarter_final_prob",
        "semi_final_prob",
        "final_prob",
        "winner_prob",
    ]
    return df[keep_cols].copy()


def main() -> None:
    elo = load_predictions("elo_poisson")
    strength = load_predictions("strength_baseline")

    merged = elo.merge(
        strength,
        on="team",
        suffixes=("_elo", "_strength"),
    )

    merged["winner_prob_diff_strength_minus_elo"] = (
        merged["winner_prob_strength"] - merged["winner_prob_elo"]
    )

    merged["final_prob_diff_strength_minus_elo"] = (
        merged["final_prob_strength"] - merged["final_prob_elo"]
    )

    merged = merged.sort_values(
        by="winner_prob_diff_strength_minus_elo",
        ascending=False,
    )

    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "tables"
        / "stage1_model_comparison_winner_probs.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print("\nStage 1 model comparison")
    print("------------------------")

    print("\nTeams rated higher by strength_baseline than elo_poisson:")
    print(
        merged[
            [
                "team",
                "winner_prob_elo",
                "winner_prob_strength",
                "winner_prob_diff_strength_minus_elo",
            ]
        ]
        .head(15)
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\nTeams rated higher by elo_poisson than strength_baseline:")
    print(
        merged[
            [
                "team",
                "winner_prob_elo",
                "winner_prob_strength",
                "winner_prob_diff_strength_minus_elo",
            ]
        ]
        .tail(15)
        .sort_values("winner_prob_diff_strength_minus_elo")
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print(f"\nSaved comparison to:")
    print(output_path)


if __name__ == "__main__":
    main()