# Results

**Selected model:** random forest (isotonic calibration)
**Operating threshold:** 0.05

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.9858 |
| PR-AUC | 0.8967 |
| Accuracy | 0.9533 |
| Precision | 0.4128 |
| Recall | 0.8824 |
| F1 | 0.5625 |
| Brier | 0.0074 |
| Log loss | 0.0336 |
| Flagged rate | 0.0727 |

Confusion matrix: TN 1,385 - FP 64 - FN 6 - TP 45.

## Model comparison

| model | pr_auc_mean | pr_auc_std | roc_auc_mean | roc_auc_std |
| --- | --- | --- | --- | --- |
| random_forest | 0.8694 | 0.0436 | 0.9712 | 0.0119 |
| hist_gradient_boosting | 0.8460 | 0.0362 | 0.9730 | 0.0121 |
| extra_trees | 0.8240 | 0.0337 | 0.9678 | 0.0036 |
| logistic_regression | 0.4700 | 0.0246 | 0.9302 | 0.0027 |

## Maintenance policy

- Threshold: 0.05
- Cost ratio: 15:1
- Expected cost per 1,000: 65
- Recall: 0.9412
- Precision: 0.4752

## Failure-mode coverage

| mode | label | occurrences | detected | recall |
| --- | --- | --- | --- | --- |
| twf | Tool wear | 8 | 2 | 0.2500 |
| hdf | Heat dissipation | 18 | 18 | 1.0000 |
| pwf | Power | 13 | 13 | 1.0000 |
| osf | Overstrain | 15 | 15 | 1.0000 |
| rnf | Random | 2 | 1 | 0.5000 |

## Risk tiers

| tier | machines | failures | failure_rate |
| --- | --- | --- | --- |
| Routine | 1342 | 3 | 0.22% |
| Monitor | 103 | 5 | 4.85% |
| Inspect | 9 | 1 | 11.11% |
| Stop | 46 | 42 | 91.30% |

## Top features

| feature | importance_mean | importance_std |
| --- | --- | --- |
| strain_index | 0.10599 | 0.00814 |
| power_w | 0.0842 | 0.01499 |
| temp_ratio | 0.0602 | 0.01361 |
| speed_deficit | 0.04497 | 0.00651 |
| type | 0.03166 | 0.02089 |
| temp_delta_k | 0.02859 | 0.00364 |
| torque_per_1000_rpm | 0.02364 | 0.00238 |
| torque_nm | 0.01916 | 0.00327 |
| rpm | 0.019 | 0.00482 |
| tool_wear_min | 0.01466 | 0.00509 |
| wear_per_power | 0.00886 | 0.00194 |
| thermal_load | 0.00392 | 0.00205 |
