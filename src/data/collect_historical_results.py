from pathlib import Path

import pandas as pd

from src.data.loaders import DATA_RAW


RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
GOALSCORERS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
)


def collect_historical_results(output_dir: Path | None = None) -> None:
    """
    Download historical international results, goal scorers, and shootouts.

    Source:
    martj42/international_results on GitHub.
    """
    output_dir = output_dir or DATA_RAW
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "historical_results.csv": RESULTS_URL,
        "historical_goalscorers.csv": GOALSCORERS_URL,
        "historical_shootouts.csv": SHOOTOUTS_URL,
    }

    for filename, url in files.items():
        print(f"Downloading {filename}...")
        df = pd.read_csv(url)
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df):,} rows to {output_path}")


if __name__ == "__main__":
    collect_historical_results()