import pandas as pd

from maintenance.data import (
    clean_frame,
    dataset_summary,
    load_raw,
    median_template,
    save_processed,
    split_data,
)


def test_load_raw_has_expected_shape(frame):
    assert len(frame) == 10000
    assert set(frame["machine_failure"].unique()) == {0, 1}
    assert {"type", "tool_wear_band", "twf", "hdf", "pwf", "osf", "rnf"}.issubset(frame.columns)


def test_invalid_type_is_normalised(sample):
    edited = sample.copy()
    edited.loc[edited.index[0], "type"] = "X"
    cleaned = clean_frame(edited)
    assert cleaned.loc[0, "type"] == "M"


def test_sensor_ranges_are_clamped(sample):
    edited = sample.copy()
    edited.loc[edited.index[0], "tool_wear_min"] = 999
    edited.loc[edited.index[0], "rpm"] = -50
    cleaned = clean_frame(edited)
    assert cleaned.loc[0, "tool_wear_min"] == 300
    assert cleaned.loc[0, "rpm"] == 0


def test_split_sizes_and_stratification(splits):
    train, validation, test = splits
    assert len(train) + len(validation) + len(test) == 10000
    assert len(validation) == len(test)
    rates = [split["machine_failure"].mean() for split in (train, validation, test)]
    assert max(rates) - min(rates) < 0.01


def test_split_is_deterministic(frame):
    first = split_data(frame)
    second = split_data(frame)
    for left, right in zip(first, second, strict=True):
        assert left.equals(right)


def test_summary_and_template(frame):
    summary = dataset_summary(frame)
    assert summary["rows"] == len(frame)
    assert 0 < summary["failure_rate"] < 1
    assert set(summary["types"]) == {"L", "M", "H"}
    assert summary["missing_values"] == 0

    template = median_template(frame)
    assert "machine_failure" not in template
    assert template["type"] == "M"
    assert isinstance(template["torque_nm"], float)


def test_save_processed_roundtrip(sample, tmp_path):
    path = save_processed(sample, tmp_path / "processed.csv")
    restored = pd.read_csv(path)
    assert len(restored) == len(sample)
    assert "machine_failure" in restored.columns


def test_load_raw_rejects_missing_file():
    try:
        load_raw("does-not-exist.csv")
    except FileNotFoundError as error:
        assert "fetch_data" in str(error)
    else:
        raise AssertionError("expected FileNotFoundError")
