"""Download the AI4I 2020 predictive-maintenance dataset.

Usage:
    python scripts/fetch_data.py [--force]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "ai4i2020.csv"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"


def fetch(force: bool = False) -> Path:
    if OUTPUT.exists() and not force:
        print(f"already present: {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
        return OUTPUT
    frame = pd.read_csv(URL)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT} shape={frame.shape}")
    print(f"failure rate={frame['Machine failure'].mean():.4f}")
    return OUTPUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    arguments = parser.parse_args()
    fetch(arguments.force)
