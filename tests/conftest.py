"""Shared fixtures for the maintenance test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from maintenance.data import load_raw, split_data


@pytest.fixture(scope="session")
def frame():
    return load_raw()


@pytest.fixture(scope="session")
def splits(frame):
    return split_data(frame)


@pytest.fixture(scope="session")
def sample(frame):
    failures = frame[frame["machine_failure"] == 1]
    healthy = frame[frame["machine_failure"] == 0].head(500)
    return (
        pd.concat([failures, healthy])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
