# Model Card - Machine Failure Classifier

## Model details

- **Name:** predictive-maintenance-pipeline
- **Version:** 0.1.0
- **Type:** random forest with isotonic calibration
- **Target:** probability of machine failure (`machine_failure = 1`)
- **Training data:** AI4I 2020, 10,000 observations, 3.39% failure rate
- **License:** MIT for code; CC BY 4.0 for the dataset

## Intended use

Condition-monitoring research, maintenance-policy simulation and decision-support prototyping for rotating machinery with similar sensors.

## Out-of-scope use

- Autonomous shutdown without human confirmation.
- Safety-critical assets without a validated failure-mode model and redundancy.
- Machines whose sensor ranges or operating regimes differ materially from AI4I.

## Inputs

Product type, air temperature, process temperature, rotational speed, torque and tool wear. Derived features include power, temperature delta, strain index and wear interactions.

## Metrics (held-out test set)

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.9858 |
| PR-AUC | 0.8967 |
| Recall | 0.8824 |
| Precision | 0.4128 |
| F1 | 0.5625 |
| Brier | 0.0074 |

## Maintenance policy

- Operating threshold: **0.05**
- Cost ratio: **15:1** (missed failure : false alarm)
- Expected cost per 1,000 machines: **65**

## Failure-mode coverage

| mode | label | occurrences | detected | recall |
| --- | --- | --- | --- | --- |
| twf | Tool wear | 8 | 2 | 0.2500 |
| hdf | Heat dissipation | 18 | 18 | 1.0000 |
| pwf | Power | 13 | 13 | 1.0000 |
| osf | Overstrain | 15 | 15 | 1.0000 |
| rnf | Random | 2 | 1 | 0.5000 |

## Limitations

- Synthetic benchmark; field performance will differ.
- Predicts any failure, not a specific failure mode.
- No temporal or cross-machine validation.
- Cost weights are illustrative.

## Maintenance

Retrain on fresh sensor data, re-check calibration and failure-mode coverage, and archive the manifest for every release.
