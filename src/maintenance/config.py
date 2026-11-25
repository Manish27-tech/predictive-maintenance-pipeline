"""Configuration for the predictive-maintenance pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw" / "ai4i2020.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "maintenance_clean.csv"
ARTIFACTS = ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS / "figures"
DASHBOARD_DIR = ARTIFACTS / "dashboard"
MODEL_PATH = ARTIFACTS / "maintenance_model.joblib"
MANIFEST_PATH = ARTIFACTS / "manifest.json"

REPORTS_DIR = ROOT / "reports"
REPORT_FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = ROOT / "docs"
DOCS_FIGURES_DIR = DOCS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
CV_FOLDS = 3
TARGET = "machine_failure"

# A missed failure is treated as 15 times costlier than an unnecessary
# inspection. The threshold is selected on the validation split.
FALSE_NEGATIVE_COST = 15.0
FALSE_POSITIVE_COST = 1.0

FAILURE_MODES = ["twf", "hdf", "pwf", "osf", "rnf"]
MODE_LABELS = {
    "twf": "Tool wear",
    "hdf": "Heat dissipation",
    "pwf": "Power",
    "osf": "Overstrain",
    "rnf": "Random",
}

RISK_TIERS = [
    ("Routine", 0.00, 0.02),
    ("Monitor", 0.02, 0.10),
    ("Inspect", 0.10, 0.30),
    ("Stop", 0.30, 1.01),
]

SENSOR_FEATURES = [
    "air_temp_k",
    "process_temp_k",
    "rpm",
    "torque_nm",
    "tool_wear_min",
]

ENGINEERED = [
    "temp_delta_k",
    "power_w",
    "strain_index",
    "torque_per_1000_rpm",
    "wear_per_power",
    "temp_ratio",
    "thermal_load",
    "speed_deficit",
]

NUMERIC_FEATURES = [*SENSOR_FEATURES, *ENGINEERED]
CATEGORICAL_FEATURES = ["type"]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]

OPERATIONAL_SLICES = ["type", "tool_wear_band"]
