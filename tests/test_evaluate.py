import numpy as np

from maintenance.evaluate import (
    calibration_table,
    curve_data,
    metrics_at_threshold,
    risk_tier,
    risk_tier_table,
    score_frame,
    threshold_sweep,
)


def test_metrics_at_threshold_has_expected_fields():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    result = metrics_at_threshold(labels, scores, 0.5)
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}
    assert 0 <= result["roc_auc"] <= 1
    assert 0 <= result["pr_auc"] <= 1
    assert result["missed_failures"] == 1


def test_threshold_sweep_is_sorted_and_bounded():
    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.05, 0.2, 0.35, 0.55, 0.8, 0.95])
    table = threshold_sweep(labels, scores)
    assert len(table) == 99
    assert table["threshold"].is_monotonic_increasing
    assert table["precision"].between(0, 1).all()
    assert table["recall"].between(0, 1).all()


def test_calibration_and_curves():
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, 500)
    scores = rng.random(500)
    table = calibration_table(labels, scores, bins=8)
    assert list(table.columns) == ["predicted", "observed"]
    curves = curve_data(labels, scores)
    assert len(curves["roc"]["fpr"]) == len(curves["roc"]["tpr"])
    assert len(curves["pr"]["precision"]) == len(curves["pr"]["recall"])


def test_risk_tiers_and_score_frame(sample):
    scores = np.linspace(0, 1, len(sample))
    tiers = risk_tier_table(sample, scores)
    assert tiers["machines"].sum() == len(sample)
    assert len(tiers) == 4
    assert risk_tier(0.0) == "Routine"
    assert risk_tier(0.5) == "Stop"
    scored = score_frame(sample, scores, 0.5)
    assert {"machine_failure", "probability", "prediction", "risk_tier", "type", "twf"}.issubset(scored.columns)
    assert scored["prediction"].isin([0, 1]).all()
