import pandas as pd

from src.data.loaders import PROJECT_ROOT
from src.visualization.plots import (
    plot_golden_boot_probabilities,
    plot_qualification_probabilities,
    plot_winner_probabilities,
)


def main() -> None:
    predictions_dir = PROJECT_ROOT / "outputs" / "predictions"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"

    winner_path = predictions_dir / "stage1_winner_predictions.csv"
    golden_boot_path = predictions_dir / "stage1_golden_boot_predictions.csv"
    group_path = predictions_dir / "group_stage_qualification_probabilities.csv"

    if not winner_path.exists():
        raise FileNotFoundError(f"Missing file: {winner_path}")

    if not golden_boot_path.exists():
        raise FileNotFoundError(f"Missing file: {golden_boot_path}")

    winner_predictions = pd.read_csv(winner_path)
    golden_boot_predictions = pd.read_csv(golden_boot_path)

    plot_winner_probabilities(
        winner_predictions=winner_predictions,
        output_path=figures_dir / "stage1_winner_probabilities.png",
    )

    plot_golden_boot_probabilities(
        golden_boot_predictions=golden_boot_predictions,
        output_path=figures_dir / "stage1_golden_boot_probabilities.png",
    )

    if group_path.exists():
        group_predictions = pd.read_csv(group_path)

        plot_qualification_probabilities(
            group_predictions=group_predictions,
            output_path=figures_dir / "group_stage_qualification_probabilities.png",
        )

    print("\nStage 1 plots created")
    print("---------------------")
    print(figures_dir / "stage1_winner_probabilities.png")
    print(figures_dir / "stage1_golden_boot_probabilities.png")

    if group_path.exists():
        print(figures_dir / "group_stage_qualification_probabilities.png")


if __name__ == "__main__":
    main()