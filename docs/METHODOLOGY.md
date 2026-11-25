# Methodology

## Data

- **Source:** AI4I 2020 predictive-maintenance dataset (UCI), 10,000 machine observations.
- **Target:** `machine_failure = 1` if the machine failed in that observation.
- **Failure modes:** tool wear (TWF), heat dissipation (HDF), power (PWF), overstrain (OSF) and random (RNF). These are multi-label and are used for diagnostics, not as model inputs.
- **Cleaning:** normalise product type, clamp sensors to physical ranges, build tool-wear bands and remove duplicates.
- **Split:** stratified 70/15/15 train/validation/test with seed 42. The validation split chooses the threshold; the test split is evaluated once.

## Physics-inspired features

- `temp_delta_k`: process temperature minus air temperature.
- `power_w`: torque x angular speed, `2 * pi * rpm / 60`.
- `strain_index`: torque x tool wear / 1000, a proxy for overstrain risk.
- `torque_per_1000_rpm`: torque normalised by speed.
- `wear_per_power`: tool wear per unit of mechanical power.
- `temp_ratio`: process/air temperature ratio.
- `thermal_load`: temperature delta x torque / 100.
- `speed_deficit`: 1500 rpm minus actual speed.

Product type is one-hot encoded. All numeric features are median-imputed and standard-scaled.

## Model selection

Four candidates are compared with 3-fold stratified cross-validation, ranked by PR-AUC:

1. Logistic regression with balanced class weights.
2. Random forest with balanced subsample weights (350 trees).
3. Extra trees with balanced weights (350 trees).
4. Histogram gradient boosting.

Random forest won with CV PR-AUC 0.8694, followed by histogram gradient boosting (0.8460), extra trees (0.8240) and logistic regression (0.4700).

## Calibration and maintenance policy

- The winning pipeline is calibrated with isotonic regression using 3 internal folds.
- The threshold is chosen on validation by minimising `15 * missed failures + 1 * false alarm` per machine.
- The resulting threshold of 0.05 is applied to the test set.
- A policy curve reports recall, precision, flagged rate, missed failures, false alarms and expected cost per 1,000 machines at every threshold from 0.01 to 0.99.

## Evaluation

- Discrimination: ROC-AUC and PR-AUC.
- Operating point: accuracy, precision, recall, F1, flagged rate, missed failures, false alarms.
- Probability quality: Brier score and log loss.
- Risk tiers: routine, monitor, inspect, stop.
- Failure-mode coverage: share of each labelled mode detected at the operating threshold.
- Operational slices: product type and tool-wear band.

## Reproducibility

```bash
python scripts/fetch_data.py
python -m maintenance.pipeline
python -m maintenance.cli check
```

All metrics, figures, predictions and the fitted model are regenerated from `data/raw/ai4i2020.csv`. Checksums are stored in `artifacts/manifest.json`.
