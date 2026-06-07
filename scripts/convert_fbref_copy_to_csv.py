from pathlib import Path
import argparse
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "club_player_stats_sources"


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def make_unique_columns(columns: list[str]) -> list[str]:
    counts = {}
    out = []

    for col in columns:
        col = str(col).strip()

        if col == "" or col.lower() == "nan":
            col = "unnamed"

        if col not in counts:
            counts[col] = 0
            out.append(col)
        else:
            counts[col] += 1
            out.append(f"{col}_{counts[col]}")

    return out

def read_fbref_copy_table(input_path: Path) -> pd.DataFrame:
    """
    Reads an FBref copied table saved as .csv/.txt.

    This is deliberately tolerant of uneven FBref copy-paste rows:
    - grouped header row may have fewer columns
    - actual header row may have many columns
    - repeated header rows can appear inside the table
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input file: {input_path}")

    try:
        text = input_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = input_path.read_text(encoding="latin1")

    lines = [line.strip("\n\r") for line in text.splitlines() if line.strip()]

    if not lines:
        raise ValueError(f"Input file is empty: {input_path}")

    # Detect delimiter. FBref copy-paste is usually tab-separated.
    tab_count = sum(line.count("\t") for line in lines[:10])
    comma_count = sum(line.count(",") for line in lines[:10])

    delimiter = "\t" if tab_count >= comma_count else ","

    rows = [line.split(delimiter) for line in lines]

    # Find the real header row: the one containing Player and usually Rk.
    header_idx = None

    for i, row in enumerate(rows[:10]):
        cleaned = [cell.strip() for cell in row]
        if "Player" in cleaned:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            "Could not find a header row containing 'Player'. "
            f"First rows were: {rows[:3]}"
        )

    header = [cell.strip() for cell in rows[header_idx]]
    data_rows = rows[header_idx + 1 :]

    width = len(header)

    normalized_rows = []

    for row in data_rows:
        row = [cell.strip() for cell in row]

        # Skip malformed short rows, such as remaining grouped-header fragments.
        if len(row) < 3:
            continue

        # Pad or truncate rows to header width.
        if len(row) < width:
            row = row + [""] * (width - len(row))
        elif len(row) > width:
            row = row[:width]

        normalized_rows.append(row)

    df = pd.DataFrame(normalized_rows, columns=make_unique_columns(header))

    # Remove repeated header rows inside FBref tables.
    if "Rk" in df.columns:
        df = df[df["Rk"].astype(str).str.strip() != "Rk"]

    if "Player" in df.columns:
        df = df[df["Player"].notna()]
        df = df[df["Player"].astype(str).str.strip() != ""]

    return df.reset_index(drop=True)


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts common FBref numeric columns while leaving text columns untouched.
    Handles duplicate columns made unique as Gls_1, Ast_1, etc.
    """
    numeric_base_names = {
        "Rk",
        "Age",
        "Born",
        "MP",
        "Starts",
        "Min",
        "90s",
        "Gls",
        "Ast",
        "G+A",
        "G-PK",
        "PK",
        "PKatt",
        "CrdY",
        "CrdR",
        "xG",
        "npxG",
        "xAG",
        "npxG+xAG",
        "PrgC",
        "PrgP",
        "PrgR",
        "Sh",
        "SoT",
        "SoT%",
        "Sh/90",
        "SoT/90",
        "G/Sh",
        "G/SoT",
        "Dist",
        "FK",
    }

    for col in df.columns:
        base_col = re.sub(r"_\d+$", "", col)

        if base_col in numeric_base_names:
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    return df


def add_model_safe_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds missing model columns so files with only standard stats do not break.
    Shooting/xG files should already contain these.
    """
    defaults = {
        "xG": 0.0,
        "npxG": 0.0,
        "xAG": 0.0,
        "Sh": 0.0,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    return df


def convert_file(
    input_path: Path,
    output_path: Path | None = None,
    source_name: str | None = None,
) -> Path:
    input_path = input_path.resolve()

    if source_name is None:
        source_name = input_path.stem

    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"{slugify(source_name)}.csv"
    else:
        output_path = output_path.resolve()

    df = read_fbref_copy_table(input_path)
    df = clean_numeric_columns(df)
    df = add_model_safe_defaults(df)

    df["source_file"] = output_path.name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print("\nConverted FBref copy table")
    print("--------------------------")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows:   {len(df):,}")
    print(f"Cols:   {len(df.columns):,}")
    print("\nColumns:")
    print(list(df.columns))
    print("\nPreview:")
    print(df.head(10).to_string(index=False))

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an FBref copied table into a clean CSV for the player-stat pipeline."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input file copied from FBref, e.g. data/external/mls.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to data/raw/club_player_stats_sources/<input_stem>.csv",
    )

    parser.add_argument(
        "--source-name",
        type=str,
        default=None,
        help="Optional source name used for default output filename.",
    )

    args = parser.parse_args()

    convert_file(
        input_path=args.input,
        output_path=args.output,
        source_name=args.source_name,
    )


if __name__ == "__main__":
    main()