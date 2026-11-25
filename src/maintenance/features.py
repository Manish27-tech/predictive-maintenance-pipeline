"""Physics-inspired feature engineering for rotating-machinery sensors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    SENSOR_FEATURES,
)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add thermal, power, strain and wear interactions."""
    out = frame.copy()
    for column in SENSOR_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["temp_delta_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["power_w"] = out["torque_nm"] * 2 * np.pi * out["rpm"] / 60
    out["strain_index"] = out["torque_nm"] * out["tool_wear_min"] / 1000
    out["torque_per_1000_rpm"] = out["torque_nm"] / (out["rpm"] / 1000 + 1)
    out["wear_per_power"] = out["tool_wear_min"] / (out["power_w"] / 1000 + 1)
    out["temp_ratio"] = out["process_temp_k"] / out["air_temp_k"]
    out["thermal_load"] = out["temp_delta_k"] * out["torque_nm"] / 100
    out["speed_deficit"] = 1500 - out["rpm"]

    out["type"] = out["type"].fillna("M").astype(str)
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    return out


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_input(frame: pd.DataFrame) -> pd.DataFrame:
    return build_features(frame)[FEATURES]


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())
