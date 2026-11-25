# Maintenance policy

## Cost model

A missed failure is treated as **15x** costlier than an unnecessary inspection:

```text
expected cost = 15 * false negatives + 1 * false positive
```

The ratio is a planning assumption, not a universal truth. It should be replaced with real downtime, repair, spare-parts and inspection costs before deployment.

## Threshold selection

1. Score every validation machine with the calibrated model.
2. Compute expected cost for thresholds from 0.01 to 0.99.
3. Choose the threshold with the lowest cost per 1,000 machines.
4. Freeze that threshold and evaluate the test set once.

The selected threshold is **0.05** on validation, with expected cost **65.33 per 1,000** and recall **94.12%**.

## Operating point on test

| Measure | Value |
| --- | ---: |
| Threshold | 0.05 |
| Recall | 88.24% |
| Precision | 41.28% |
| Flagged rate | 7.27% |
| Missed failures | 6 of 51 |
| False alarms | 64 |
| Expected cost | 154 units |

The test-optimal threshold would be 0.08 (cost 72 units), but it is not used because tuning on test would leak information. The gap between validation and test is a useful reminder that the policy is estimated from a finite sample.

## Actions by risk tier

| Tier | Probability | Action |
| --- | --- | --- |
| Routine | 0-2% | Continue normal operation |
| Monitor | 2-10% | Schedule condition monitoring |
| Inspect | 10-30% | Inspect before next shift |
| Stop | 30%+ | Stop and service immediately |

## Known gap: tool-wear failures

At the operating threshold, the model catches 100% of heat-dissipation, power and overstrain failures, 50% of random failures and only 25% of tool-wear failures. Tool-wear failures are rare and can look healthy until the wear limit is reached.

**Recommended mitigation:** add a hard inspection rule at high tool wear (for example, 200+ minutes) until a dedicated tool-wear model is validated. Do not silently rely on the classifier for that mode.

## Review cadence

- Re-estimate the cost ratio every quarter.
- Re-check failure-mode coverage after every retraining.
- Track calibration and score distribution by product type.
- Revalidate the hard tool-wear rule with maintenance records.
