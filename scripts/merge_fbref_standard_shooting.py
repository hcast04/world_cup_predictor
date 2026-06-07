from pathlib import Path
import argparse

import pandas as pd


def clean_key(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(".", "")
        .replace("-", " ")
        .replace("'", "")
        .replace("’", "")
    )


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    for col in df.columns:
        if col not in {"Player", "Nation", "Pos", "Squad", "Matches", "source_file"}:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="ignore",
            )

    return df


def merge_standard_shooting(
    standard_path: Path,
    shooting_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    standard = load_table(standard_path)
    shooting = load_table(shooting_path)

    required_standard = {"Player", "Nation", "Pos", "Squad"}
    required_shooting = {"Player", "Nation", "Pos", "Squad"}

    missing_standard = required_standard - set(standard.columns)
    missing_shooting = required_shooting - set(shooting.columns)

    if missing_standard:
        raise ValueError(f"Standard file missing columns: {missing_standard}")

    if missing_shooting:
        raise ValueError(f"Shooting file missing columns: {missing_shooting}")

    for df in [standard, shooting]:
        df["player_key"] = df["Player"].map(clean_key)
        df["squad_key"] = df["Squad"].map(clean_key)

    # Keep useful shooting columns only.
    shooting_cols = [
        "player_key",
        "squad_key",
        "Sh",
        "SoT",
        "SoT%",
        "Sh/90",
        "SoT/90",
        "G/Sh",
        "G/SoT",
    ]

    shooting_cols = [col for col in shooting_cols if col in shooting.columns]

    shooting_small = shooting[shooting_cols].copy()

    merged = standard.merge(
        shooting_small,
        on=["player_key", "squad_key"],
        how="left",
        suffixes=("", "_shooting"),
    )

    # If shooting columns were already present in standard, prefer explicit shooting-table values.
    for col in ["Sh", "SoT", "SoT%", "Sh/90", "SoT/90", "G/Sh", "G/SoT"]:
        shooting_col = f"{col}_shooting"

        if shooting_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[shooting_col].combine_first(merged[col])
            else:
                merged[col] = merged[shooting_col]

            merged = merged.drop(columns=[shooting_col])

    # FBref standard table has no xG. Keep safe defaults.
    # Later, if you copy FBref expected/shooting xG tables, we can merge real xG.
    for col in ["xG", "npxG", "xAG"]:
        if col not in merged.columns:
            merged[col] = 0.0

    merged = merged.drop(columns=["player_key", "squad_key"])

    merged["source_file"] = output_path.name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print("\nMerged FBref standard + shooting")
    print("--------------------------------")
    print(f"Standard input: {standard_path}")
    print(f"Shooting input: {shooting_path}")
    print(f"Output:         {output_path}")
    print(f"Rows:           {len(merged):,}")
    print(f"Columns:        {len(merged.columns):,}")

    print("\nColumns:")
    print(list(merged.columns))

    print("\nPreview:")
    preview_cols = [
        col
        for col in [
            "Player",
            "Nation",
            "Pos",
            "Squad",
            "MP",
            "Starts",
            "Min",
            "90s",
            "Gls",
            "Ast",
            "G-PK",
            "PK",
            "PKatt",
            "Sh",
            "SoT",
            "Sh/90",
            "SoT/90",
            "xG",
            "npxG",
            "xAG",
        ]
        if col in merged.columns
    ]

    print(merged[preview_cols].head(20).to_string(index=False))

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge FBref standard and shooting player tables into one CSV."
    )

    parser.add_argument("--standard", type=Path, required=True)
    parser.add_argument("--shooting", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    merge_standard_shooting(
        standard_path=args.standard,
        shooting_path=args.shooting,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()