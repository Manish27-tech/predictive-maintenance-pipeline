# Predictive Maintenance Pipeline

[![CI](https://github.com/Manish27-tech/predictive-maintenance-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Manish27-tech/predictive-maintenance-pipeline/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

An end-to-end, cost-sensitive machine-failure pipeline on the AI4I 2020 predictive-maintenance benchmark. It cleans sensor data, engineers physics-inspired features, compares four classifiers, calibrates the winner and converts probabilities into a maintenance policy with explicit missed-failure and false-alarm costs.

**Live results:** https://manish27-tech.github.io/predictive-maintenance-pipeline/

## Results

| Metric | Value |
| --- | ---: |
| Selected model | Random forest |
| ROC-AUC | 0.9858 |
| PR-AUC | 0.8967 |
| Accuracy | 0.9533 |
| Precision | 0.4128 |
| Recall | 0.8824 |
| F1 | 0.5625 |
| Brier score | 0.0074 |
| Operating threshold | 0.05 |
| Flagged rate | 7.27% |
| Missed failures | 6 of 51 |
| False alarms | 64 |

Risk tiers on the held-out test set:

| Tier | Machines | Failures | Observed failure rate |
| --- | ---: | ---: | ---: |
| Routine | 1,342 | 3 | 0.22% |
| Monitor | 103 | 5 | 4.85% |
| Inspect | 9 | 1 | 11.11% |
| Stop | 46 | 42 | 91.30% |

## Why this project

Predictive maintenance is not only a classification problem. A missed failure can stop a line, while an unnecessary inspection consumes scarce technician time. This repository makes that trade-off explicit: calibrated probabilities, a cost-sensitive threshold, a policy curve and per-failure-mode coverage.

## Architecture

```text
data/raw/ai4i2020.csv
        |
        v
clean_frame -> split_data -> physics-inspired features
        |
        v
select_model (3-fold CV: logistic, random forest, extra trees, HGB)
        |
        v
calibrate_model (isotonic) -> validation cost policy
        |
        v
test evaluation -> failure-mode coverage -> operational slices
        |
        v
artifacts/  reports/  docs/  dashboard/
```

## Project layout

```text
predictive-maintenance-pipeline/
  src/maintenance/       data, features, models, evaluation, policy, plots, reports, pipeline
  dashboard/app.py       Streamlit app with a policy simulator and live machine assessment
  docs/                  GitHub Pages site, methodology, maintenance policy, data dictionary
  reports/               Markdown report and model card
  artifacts/             metrics, figures, predictions, manifest, fitted model
  scripts/fetch_data.py  reproducible AI4I download
  tests/                 30 tests covering data, features, models, evaluation, policy and manifest
```

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
python -m maintenance.pipeline
streamlit run dashboard/app.py
```

The raw dataset is committed, so the pipeline runs offline. `python scripts/fetch_data.py` re-downloads it for provenance.

## Reproduce the published numbers

```bash
python -m maintenance.pipeline
python -m maintenance.cli check
ruff check src tests scripts dashboard
pytest
```

Every metric, figure and prediction is regenerated from `data/raw/ai4i2020.csv`. Checksums are stored in `artifacts/manifest.json`.

## Maintenance policy

A missed failure is weighted **15x** more heavily than a false alarm. The validation split chooses the threshold; the test split is used once.

| Threshold | Recall | Precision | Flagged | Cost / 1,000 |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 (chosen) | 94.12% | 47.52% | 6.73% | 65.33 |
| 0.08 (test-optimal) | 88.24% | 71.43% | 4.20% | 72.00 |

The chosen policy is deliberately recall-heavy. The test-optimal threshold is reported for transparency but is not used.

## Failure-mode coverage

| Mode | Occurrences | Detected | Recall |
| --- | ---: | ---: | ---: |
| Tool wear | 8 | 2 | 25.00% |
| Heat dissipation | 18 | 18 | 100.00% |
| Power | 13 | 13 | 100.00% |
| Overstrain | 15 | 15 | 100.00% |
| Random | 2 | 1 | 50.00% |

**Known gap:** tool-wear failures are under-detected. Add a hard inspection rule at high tool wear until a dedicated tool-wear model is validated. See [docs/MAINTENANCE_POLICY.md](docs/MAINTENANCE_POLICY.md).

## Key design choices

- **Random forest** won the 3-fold CV on PR-AUC (0.8694 vs 0.8460 for histogram gradient boosting).
- **Isotonic calibration** makes probabilities usable for risk tiers and maintenance actions.
- **15:1 cost ratio** reflects the higher cost of a missed failure; the threshold lands at 0.05.
- **Failure-mode coverage** is reported separately because aggregate recall hides the tool-wear gap.
- **Raw data is committed** so the project runs without network access.

## Dataset

AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository. 10,000 observations, 3.39% failure rate, five labelled failure modes. Distributed under CC BY 4.0.

> Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository.

## License

Code is released under the MIT license. See [LICENSE](LICENSE).
