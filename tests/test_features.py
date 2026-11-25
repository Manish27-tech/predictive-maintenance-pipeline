import numpy as np

from maintenance.config import ENGINEERED, FEATURES, NUMERIC_FEATURES
from maintenance.features import build_features, build_preprocessor, feature_names, model_input


def test_engineered_columns_are_created(sample):
    enriched = build_features(sample)
    assert set(ENGINEERED).issubset(enriched.columns)
    assert enriched[ENGINEERED].notna().all().all()


def test_physics_features_are_sensible(sample):
    enriched = build_features(sample)
    assert (enriched["power_w"] >= 0).all()
    assert (enriched["temp_delta_k"] >= 0).all()
    assert (enriched["strain_index"] >= 0).all()


def test_no_infinite_values(sample):
    enriched = build_features(sample)
    numeric = enriched[NUMERIC_FEATURES].to_numpy(dtype=float)
    assert np.isfinite(numeric).all()


def test_model_input_columns_are_stable(sample):
    inputs = model_input(sample)
    assert list(inputs.columns) == FEATURES
    assert len(inputs) == len(sample)


def test_preprocessor_output_shape(sample):
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(model_input(sample))
    assert transformed.shape[0] == len(sample)
    assert transformed.shape[1] >= len(NUMERIC_FEATURES)
    assert len(feature_names(preprocessor)) == transformed.shape[1]


def test_preprocessor_handles_unseen_type(sample):
    preprocessor = build_preprocessor()
    preprocessor.fit(model_input(sample))
    edited = sample.head(3).copy()
    edited["type"] = "Z"
    transformed = preprocessor.transform(model_input(edited))
    assert transformed.shape[0] == 3
    assert np.isfinite(transformed).all()
