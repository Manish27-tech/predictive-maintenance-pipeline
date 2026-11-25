# Architecture decisions

## 1. Model family

**Decision:** random forest as the default model.
**Reason:** it won the 3-fold PR-AUC comparison (0.8694) and is robust on a small, imbalanced sensor table. Histogram gradient boosting was close (0.8460) and remains a useful challenger.

## 2. Calibration

**Decision:** isotonic calibration.
**Reason:** risk tiers and maintenance actions depend on probability accuracy. The Brier score on test is 0.0074.

## 3. Cost ratio

**Decision:** weight a missed failure 15x more than a false alarm.
**Reason:** unplanned downtime and collateral damage dominate inspection cost. The ratio is configurable and documented in the maintenance policy.

## 4. Threshold selection

**Decision:** select on validation, evaluate once on test.
**Reason:** test-set tuning would leak information. The test-optimal threshold (0.08) is reported but not used.

## 5. Failure-mode coverage

**Decision:** report per-mode recall separately.
**Reason:** aggregate recall hides the tool-wear gap. Maintenance planners need to know which failure types the classifier misses.

## 6. Synthetic benchmark

**Decision:** use AI4I 2020 and document its limits.
**Reason:** it is public, reproducible and small enough for CI, but it is generated from known failure rules. Field validation is required before deployment.

## 7. Raw data in version control

**Decision:** commit the 0.5 MB raw CSV.
**Reason:** the project runs offline and every published number is reproducible.
