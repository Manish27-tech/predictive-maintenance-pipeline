import itertools

from maintenance.config import (
    CATEGORICAL_FEATURES,
    ENGINEERED,
    FEATURES,
    NUMERIC_FEATURES,
    RISK_TIERS,
)


def test_features_are_unique_and_disjoint():
    assert len(FEATURES) == len(set(FEATURES))
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)


def test_engineered_block_is_complete():
    assert len(ENGINEERED) == 8
    assert set(ENGINEERED).issubset(NUMERIC_FEATURES)
    assert CATEGORICAL_FEATURES == ["type"]


def test_risk_tiers_are_contiguous():
    assert RISK_TIERS[0][1] == 0.0
    assert RISK_TIERS[-1][2] >= 1.0
    for previous, current in itertools.pairwise(RISK_TIERS):
        assert previous[2] == current[1]
