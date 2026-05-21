import pandas as pd

from src.data.loaders import PROJECT_ROOT


def main() -> None:
    path = PROJECT_ROOT / "outputs" / "predictions" / "group_stage_qualification_probabilities.csv"

    df = pd.read_csv(path)

    print("\nTop qualification probabilities")
    print("-------------------------------")
    print(
        df.sort_values("qualify_prob", ascending=False)
        .head(20)
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\nLowest qualification probabilities")
    print("----------------------------------")
    print(
        df.sort_values("qualify_prob", ascending=True)
        .head(20)
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    if "finish_1_prob" in df.columns:
        print("\nMost likely group winners")
        print("-------------------------")
        print(
            df.sort_values("finish_1_prob", ascending=False)
            .head(20)
            .to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )


if __name__ == "__main__":
    main()