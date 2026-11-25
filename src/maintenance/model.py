"""Candidate models, cross-validated selection and calibration."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from .config import CV_FOLDS, RANDOM_STATE, TARGET
from .features import build_preprocessor, model_input


def candidate_models() -> dict:
    """Four model families spanning linear, bagged and boosted trees."""
    return {
        "logistic_regression": LogisticRegression(
            C=0.5,
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.2,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def select_model(
    train: pd.DataFrame,
    folds: int = CV_FOLDS,
    seed: int = RANDOM_STATE,
) -> tuple[str, Pipeline, pd.DataFrame]:
    inputs = model_input(train)
    labels = train[TARGET].to_numpy()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    rows = []
    fitted = {}
    for name, estimator in candidate_models().items():
        pipeline = build_pipeline(estimator)
        scores = cross_validate(
            pipeline,
            inputs,
            labels,
            cv=splitter,
            scoring=("average_precision", "roc_auc"),
            n_jobs=1,
        )
        rows.append(
            {
                "model": name,
                "pr_auc_mean": float(scores["test_average_precision"].mean()),
                "pr_auc_std": float(scores["test_average_precision"].std()),
                "roc_auc_mean": float(scores["test_roc_auc"].mean()),
                "roc_auc_std": float(scores["test_roc_auc"].std()),
            }
        )
        fitted[name] = pipeline.fit(inputs, labels)

    comparison = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)
    best = str(comparison.loc[0, "model"])
    return best, fitted[best], comparison


def calibrate_model(model: Pipeline, train: pd.DataFrame, folds: int = 3) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(model, cv=folds, method="isotonic")
    calibrated.fit(model_input(train), train[TARGET].to_numpy())
    return calibrated


def save_model(model, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path):
    return joblib.load(path)
