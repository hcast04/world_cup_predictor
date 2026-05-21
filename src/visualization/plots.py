from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_probability_bar_chart(
    df: pd.DataFrame,
    label_column: str,
    probability_column: str,
    title: str,
    output_path: Path,
    n_top: int = 15,
) -> None:
    """
    Create a horizontal bar chart for probability outputs.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = (
        df[[label_column, probability_column]]
        .dropna()
        .sort_values(probability_column, ascending=False)
        .head(n_top)
        .sort_values(probability_column, ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(plot_df[label_column], plot_df[probability_column] * 100)

    ax.set_xlabel("Probability (%)")
    ax.set_title(title)

    for i, value in enumerate(plot_df[probability_column] * 100):
        ax.text(value, i, f" {value:.1f}%", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_winner_probabilities(
    winner_predictions: pd.DataFrame,
    output_path: Path,
    n_top: int = 15,
) -> None:
    plot_probability_bar_chart(
        df=winner_predictions,
        label_column="team",
        probability_column="winner_prob",
        title="World Cup Winner Probabilities",
        output_path=output_path,
        n_top=n_top,
    )


def plot_golden_boot_probabilities(
    golden_boot_predictions: pd.DataFrame,
    output_path: Path,
    n_top: int = 15,
) -> None:
    plot_probability_bar_chart(
        df=golden_boot_predictions,
        label_column="player",
        probability_column="golden_boot_prob",
        title="Golden Boot Probabilities",
        output_path=output_path,
        n_top=n_top,
    )


def plot_qualification_probabilities(
    group_predictions: pd.DataFrame,
    output_path: Path,
    n_top: int = 24,
) -> None:
    plot_probability_bar_chart(
        df=group_predictions,
        label_column="team",
        probability_column="qualify_prob",
        title="Group-Stage Qualification Probabilities",
        output_path=output_path,
        n_top=n_top,
    )