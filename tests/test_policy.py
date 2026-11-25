import numpy as np

from maintenance.evaluate import score_frame
from maintenance.policy import (
    action_for_tier,
    choose_policy,
    failure_mode_recall,
    operational_slices,
    policy_curve,
)


def test_policy_curve_is_complete():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, 800)
    scores = np.clip(labels * 0.35 + rng.random(800) * 0.5, 0, 1)
    curve = policy_curve(labels, scores)
    assert len(curve) == 99
    assert curve["cost_per_1000"].min() <= curve["cost_per_1000"].max()
    assert (curve["missed_failures"] >= 0).all()


def test_choose_policy_returns_best_row():
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 2, 500)
    scores = rng.random(500)
    curve = policy_curve(labels, scores)
    chosen = choose_policy(curve)
    assert chosen["cost_per_1000"] == curve["cost_per_1000"].min()
    assert 0 < chosen["threshold"] < 1


def test_failure_mode_recall_counts(sample):
    predictions = np.ones(len(sample), dtype=int)
    table = failure_mode_recall(sample, predictions)
    assert len(table) == 5
    assert table["detected"].equals(table["occurrences"])
    assert (table["recall"].dropna() == 1.0).all()


def test_operational_slices(sample):
    scores = np.linspace(0, 1, len(sample))
    scored = score_frame(sample, scores, 0.3)
    table = operational_slices(scored, "type")
    assert table["machines"].sum() == len(sample)
    assert {"flag_rate", "recall", "precision", "mean_score"}.issubset(table.columns)


def test_action_mapping():
    assert action_for_tier("Stop").startswith("Stop")
    assert action_for_tier("Routine").startswith("Continue")
    assert action_for_tier("unknown").startswith("Inspect")
