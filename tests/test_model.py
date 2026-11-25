import numpy as np
import pandas as pd
import pytest

from maintenance.config import TARGET
from maintenance.features import model_input
from maintenance.model import (
    build_pipeline,
    calibrate_model,
    candidate_models,
    load_model,
    save_model,
    select_model,
)


def test_candidate_models_cover_four_families():
    assert set(candidate_models()) == {
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    }


def test_pipeline_fit_predict(sample):
    pipeline = build_pipeline(candidate_models()["logistic_regression"])
    pipeline.fit(model_input(sample), sample[TARGET])
    probabilities = pipeline.predict_proba(model_input(sample))
    assert probabilities.shape == (len(sample), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_select_model_ranks_by_pr_auc(sample):
    best, model, comparison = select_model(sample, folds=2, seed=13)
    assert best == comparison.iloc[0]["model"]
    assert len(comparison) == 4
    assert comparison["pr_auc_mean"].is_monotonic_decreasing
    assert model.predict_proba(model_input(sample)).shape[0] == len(sample)


def test_calibration_and_persistence(sample, tmp_path):
    _, model, _ = select_model(sample, folds=2, seed=13)
    calibrated = calibrate_model(model, sample, folds=2)
    before = calibrated.predict_proba(model_input(sample))[:, 1]
    path = save_model(calibrated, tmp_path / "model.joblib")
    restored = load_model(path)
    after = restored.predict_proba(model_input(sample))[:, 1]
    assert np.allclose(before, after)
    assert restored.predict_proba(pd.DataFrame(model_input(sample).head(2))).shape == (2, 2)


def test_save_model_creates_parent_directory(sample, tmp_path):
    pipeline = build_pipeline(candidate_models()["logistic_regression"])
    pipeline.fit(model_input(sample), sample[TARGET])
    path = save_model(pipeline, tmp_path / "nested" / "model.joblib")
    assert path.exists()
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")
