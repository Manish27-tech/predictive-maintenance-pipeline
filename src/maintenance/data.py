"""Load, clean and split the AI4I 2020 predictive-maintenance dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_RAW, FAILURE_MODES, RANDOM_STATE, TARGET, TEST_SIZE, VAL_SIZE

RENAME = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": TARGET,
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

SENSOR_COLUMNS = [
    "air_temp_k",
    "process_temp_k",
    "rpm",
    "torque_nm",
    "tool_wear_min",
]


def load_raw(path: str | Path = DATA_RAW) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found at {path}. Run `python scripts/fetch_data.py` first."
        )
    frame = pd.read_csv(path).rename(columns=RENAME)
    return clean_frame(frame)


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate sensor ranges, normalise product type and build audit bands."""
    out = frame.copy()
    if TARGET not in out.columns:
        raise ValueError(f"target column {TARGET!r} is missing")

    out = out.drop_duplicates().reset_index(drop=True)
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce").fillna(0).astype(int)
    out = out[out[TARGET].isin([0, 1])]

    for column in [*SENSOR_COLUMNS, *FAILURE_MODES]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["type"] = out["type"].astype(str).str.upper().str[0]
    out["type"] = out["type"].where(out["type"].isin(["L", "M", "H"]), "M")

    out["air_temp_k"] = out["air_temp_k"].clip(250, 350)
    out["process_temp_k"] = out["process_temp_k"].clip(250, 400)
    out["rpm"] = out["rpm"].clip(0, 4000)
    out["torque_nm"] = out["torque_nm"].clip(0, 200)
    out["tool_wear_min"] = out["tool_wear_min"].clip(0, 300)

    out["tool_wear_band"] = pd.cut(
        out["tool_wear_min"],
        bins=[-1, 50, 120, 180, 300],
        labels=["0-50", "51-120", "121-180", "181-300"],
    ).astype(str)
    return out.reset_index(drop=True)


def split_data(
    frame: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, remainder = train_test_split(
        frame,
        test_size=test_size + val_size,
        random_state=seed,
        stratify=frame[TARGET],
    )
    validation, test = train_test_split(
        remainder,
        test_size=test_size / (test_size + val_size),
        random_state=seed,
        stratify=remainder[TARGET],
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def dataset_summary(frame: pd.DataFrame) -> dict:
    mode_counts = {
        mode: int(frame[mode].sum())
        for mode in FAILURE_MODES
        if mode in frame.columns
    }
    return {
        "rows": len(frame),
        "failures": int(frame[TARGET].sum()),
        "healthy": int((1 - frame[TARGET]).sum()),
        "failure_rate": round(float(frame[TARGET].mean()), 4),
        "types": frame["type"].value_counts().to_dict(),
        "failure_modes": mode_counts,
        "mean_tool_wear": round(float(frame["tool_wear_min"].mean()), 1),
        "mean_torque": round(float(frame["torque_nm"].mean()), 1),
        "mean_rpm": round(float(frame["rpm"].mean()), 1),
        "missing_values": int(frame.isna().sum().sum()),
    }


def median_template(frame: pd.DataFrame) -> dict:
    template = {}
    for column in frame.columns:
        if column == TARGET:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            value = frame[column].median()
            template[column] = None if pd.isna(value) else float(value)
        else:
            mode = frame[column].mode()
            template[column] = str(mode.iloc[0]) if not mode.empty else "Unknown"
    template["type"] = "M"
    template["tool_wear_band"] = "51-120"
    return template


def save_processed(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def ensure_finite(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)
