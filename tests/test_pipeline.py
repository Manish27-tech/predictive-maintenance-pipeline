import numpy as np

from maintenance.pipeline import _jsonable, _split_summary


def test_split_summary_counts_match(splits):
    train, validation, test = splits
    summary = _split_summary(train, validation, test)
    assert summary["train_rows"] == len(train)
    assert summary["val_rows"] == len(validation)
    assert summary["test_rows"] == len(test)
    assert summary["train_defaults"] + summary["val_defaults"] + summary["test_defaults"] == (
        train["machine_failure"].sum()
        + validation["machine_failure"].sum()
        + test["machine_failure"].sum()
    )


def test_jsonable_converts_numpy_and_nan():
    payload = {
        "integer": np.int64(4),
        "floating": np.float64(0.5),
        "nan": float("nan"),
        "nested": [np.bool_(True), {"value": np.float32(1.25)}],
    }
    cleaned = _jsonable(payload)
    assert cleaned["integer"] == 4
    assert cleaned["floating"] == 0.5
    assert cleaned["nan"] is None
    assert cleaned["nested"][0] is True
    assert cleaned["nested"][1]["value"] == 1.25
