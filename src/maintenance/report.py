"""Markdown reports, model card and static results site."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _pct(value: float) -> str:
    return f"{100 * float(value):.2f}%"


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    view = frame[columns] if columns else frame
    header = "| " + " | ".join(str(column) for column in view.columns) + " |"
    divider = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _comparison_table(comparison: pd.DataFrame) -> str:
    view = comparison.copy()
    for column in ["pr_auc_mean", "pr_auc_std", "roc_auc_mean", "roc_auc_std"]:
        view[column] = view[column].map(lambda value: f"{value:.4f}")
    return _markdown_table(view)


def _mode_table(modes: pd.DataFrame) -> str:
    view = modes.copy()
    view["recall"] = view["recall"].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return _markdown_table(view)


def _slice_table(slices: pd.DataFrame) -> str:
    view = slices[["slice", "group", "machines", "failures", "flag_rate", "recall", "precision", "mean_score"]].copy()
    for column in ["flag_rate", "recall", "precision", "mean_score"]:
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return _markdown_table(view)


def _tier_table(tiers: pd.DataFrame) -> str:
    view = tiers.copy()
    view["failure_rate"] = view["failure_rate"].map(lambda value: _pct(value))
    return _markdown_table(view)


def write_report(
    *,
    summary: dict,
    split_summary: dict,
    model_name: str,
    comparison: pd.DataFrame,
    policy: dict,
    test_optimal: dict,
    metrics: dict,
    modes: pd.DataFrame,
    slices: pd.DataFrame,
    tiers: pd.DataFrame,
    importance: pd.DataFrame,
    paths: dict[str, Path],
) -> Path:
    top = importance.head(8)
    report = f"""# Predictive Maintenance Pipeline

**Dataset:** AI4I 2020 predictive-maintenance benchmark - {summary['rows']:,} machine observations, {summary['failures']:,} failures ({_pct(summary['failure_rate'])}).
**Task:** flag an imminent machine failure from air/process temperature, rotational speed, torque and tool wear.
**Operating model:** {model_name.replace('_', ' ')} with isotonic calibration and a cost-sensitive maintenance threshold.

## 1. Executive summary

- Test-set **ROC-AUC {metrics['roc_auc']:.4f}**, **PR-AUC {metrics['pr_auc']:.4f}**, **recall {_pct(metrics['recall'])}** and **precision {_pct(metrics['precision'])}**.
- At the chosen threshold of **{metrics['threshold']:.2f}**, **{_pct(metrics['flagged_rate'])}** of machines are flagged for inspection and **{metrics['missed_failures']}** of {split_summary['test_defaults']} failures are missed.
- The policy minimises `{policy['fn_cost']:.0f} x missed failure + {policy['fp_cost']:.0f} x false alarm`, giving an expected **{policy['cost_per_1000']:.0f} cost units per 1,000 machines** on validation.
- Failure-mode coverage, product-type slices and calibrated risk tiers are reported so maintenance planners can see where the model is weak.

## 2. Dataset and splits

| Split | Machines | Failures | Failure rate |
| --- | ---: | ---: | ---: |
| Train | {split_summary['train_rows']:,} | {split_summary['train_defaults']:,} | {_pct(split_summary['train_rate'])} |
| Validation | {split_summary['val_rows']:,} | {split_summary['val_defaults']:,} | {_pct(split_summary['val_rate'])} |
| Test | {split_summary['test_rows']:,} | {split_summary['test_defaults']:,} | {_pct(split_summary['test_rate'])} |

The split is stratified 70/15/15 with seed 42. The validation split selects the threshold; the test split is used once for final reporting. Duplicate rows are removed before splitting.

## 3. Method

1. **Cleaning:** normalise product type, validate sensor ranges, build tool-wear bands and remove duplicates.
2. **Physics-inspired features:** temperature delta, mechanical power, torque-wear strain index, torque per 1,000 rpm, wear per unit power, process/air temperature ratio, thermal load and speed deficit.
3. **Preprocessing:** median imputation and standard scaling for sensors; one-hot encoding for product type.
4. **Model selection:** 3-fold stratified CV over logistic regression, random forest, extra trees and histogram gradient boosting, ranked by PR-AUC.
5. **Calibration:** isotonic calibration of the winning pipeline.
6. **Maintenance policy:** choose the threshold on validation that minimises expected cost with a {policy['fn_cost']:.0f}:{policy['fp_cost']:.0f} missed-failure-to-false-alarm ratio.

### Model comparison

{_comparison_table(comparison)}

## 4. Test-set performance

| Metric | Value |
| --- | ---: |
| ROC-AUC | {metrics['roc_auc']:.4f} |
| PR-AUC | {metrics['pr_auc']:.4f} |
| Accuracy | {_pct(metrics['accuracy'])} |
| Precision | {_pct(metrics['precision'])} |
| Recall | {_pct(metrics['recall'])} |
| F1 | {metrics['f1']:.4f} |
| Brier score | {metrics['brier']:.4f} |
| Log loss | {metrics['log_loss']:.4f} |
| Flagged rate | {_pct(metrics['flagged_rate'])} |
| Expected cost | {metrics['cost']:.0f} |

Confusion matrix: TN {metrics['confusion_matrix']['tn']:,}, FP {metrics['confusion_matrix']['fp']:,}, FN {metrics['confusion_matrix']['fn']:,}, TP {metrics['confusion_matrix']['tp']:,}.

The validation policy chose **{policy['threshold']:.2f}** with expected cost **{policy['cost_per_1000']:.0f} per 1,000**. The test-optimal threshold would be {test_optimal['threshold']:.2f} with cost {test_optimal['cost_per_1000']:.0f}; the difference is reported instead of retuning on test.

![ROC and PR curves](figures/06_roc_pr.png)

![Calibration](figures/07_calibration.png)

## 5. Maintenance policy

The policy curve translates every threshold into missed failures, false alarms and expected cost. A missed failure is {policy['fn_cost']:.0f}x costlier than an unnecessary inspection. The selected threshold is a business decision, not only a statistical one.

![Policy curve](figures/09_policy_curve.png)

## 6. Risk tiers

{_tier_table(tiers)}

![Risk tiers](figures/13_risk_tiers.png)

## 7. Failure-mode coverage

{_mode_table(modes)}

![Failure-mode recall](figures/11_failure_mode_recall.png)

## 8. Operational slices

{_slice_table(slices)}

![Product-type performance](figures/12_operational_slices.png)

## 9. Feature importance

{_markdown_table(top[['feature', 'importance_mean', 'importance_std']])}

![Feature importance](figures/10_feature_importance.png)

## 10. Limitations

- AI4I is a synthetic benchmark built around known failure rules; real sensor data is noisier and has different failure modes.
- The cost ratio is illustrative and should be replaced with downtime, repair and inspection economics.
- Only 10,000 observations are available, so subgroup estimates for rare failure modes are uncertain.
- The model predicts any failure, not the specific failure mode; maintenance crews still need diagnosis.
- No temporal validation: rows are shuffled, so sensor drift over time is not tested.

## 11. Reproducibility

```bash
python scripts/fetch_data.py
python -m maintenance.pipeline
python -m maintenance.cli check
streamlit run dashboard/app.py
```

All metrics, figures, predictions and the fitted model are regenerated from `data/raw/ai4i2020.csv`. Checksums are in `artifacts/manifest.json`.

## 12. Next steps

1. Add an out-of-time split and drift monitoring.
2. Train a multi-label model for the five failure modes.
3. Add remaining-useful-life regression on run-to-failure histories.
4. Calibrate the cost model with real maintenance records.
"""
    path = paths["reports"] / "REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path


def write_model_card(
    *,
    summary: dict,
    model_name: str,
    metrics: dict,
    policy: dict,
    modes: pd.DataFrame,
    paths: dict[str, Path],
) -> Path:
    card = f"""# Model Card - Machine Failure Classifier

## Model details

- **Name:** predictive-maintenance-pipeline
- **Version:** 0.1.0
- **Type:** {model_name.replace('_', ' ')} with isotonic calibration
- **Target:** probability of machine failure (`machine_failure = 1`)
- **Training data:** AI4I 2020, {summary['rows']:,} observations, {summary['failure_rate']:.2%} failure rate
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
| ROC-AUC | {metrics['roc_auc']:.4f} |
| PR-AUC | {metrics['pr_auc']:.4f} |
| Recall | {metrics['recall']:.4f} |
| Precision | {metrics['precision']:.4f} |
| F1 | {metrics['f1']:.4f} |
| Brier | {metrics['brier']:.4f} |

## Maintenance policy

- Operating threshold: **{policy['threshold']:.2f}**
- Cost ratio: **{policy['fn_cost']:.0f}:{policy['fp_cost']:.0f}** (missed failure : false alarm)
- Expected cost per 1,000 machines: **{policy['cost_per_1000']:.0f}**

## Failure-mode coverage

{_mode_table(modes)}

## Limitations

- Synthetic benchmark; field performance will differ.
- Predicts any failure, not a specific failure mode.
- No temporal or cross-machine validation.
- Cost weights are illustrative.

## Maintenance

Retrain on fresh sensor data, re-check calibration and failure-mode coverage, and archive the manifest for every release.
"""
    path = paths["reports"] / "MODEL_CARD.md"
    path.write_text(card, encoding="utf-8")
    return path


def write_site(
    *,
    summary: dict,
    model_name: str,
    metrics: dict,
    comparison: pd.DataFrame,
    policy: dict,
    modes: pd.DataFrame,
    tiers: pd.DataFrame,
    importance: pd.DataFrame,
    paths: dict[str, Path],
) -> Path:
    comparison_rows = "".join(
        f"<tr><td>{row.model.replace('_', ' ')}</td><td>{row.pr_auc_mean:.4f}</td><td>{row.roc_auc_mean:.4f}</td></tr>"
        for row in comparison.itertuples()
    )
    mode_rows = "".join(
        f"<tr><td>{row.label}</td><td>{row.occurrences}</td><td>{row.detected}</td><td>{'' if pd.isna(row.recall) else f'{row.recall:.1%}'}</td></tr>"
        for row in modes.itertuples()
    )
    tier_rows = "".join(
        f"<tr><td>{row.tier}</td><td>{row.machines:,}</td><td>{row.failures:,}</td><td>{row.failure_rate:.2%}</td></tr>"
        for row in tiers.itertuples()
    )
    importance_rows = "".join(
        f"<tr><td>{row.feature}</td><td>{row.importance_mean:.5f}</td><td>{row.importance_std:.5f}</td></tr>"
        for row in importance.head(12).itertuples()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predictive Maintenance Pipeline - Results</title>
<style>
:root {{ color-scheme: light; --ink:#17212b; --muted:#5c6b7a; --accent:#1f6f8b; --bg:#f6f8fa; --card:#fff; --line:#dce3ea; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; color:var(--ink); background:var(--bg); line-height:1.55; }}
header {{ background:linear-gradient(135deg,#123b4a,#1f6f8b); color:#fff; padding:56px 24px; }}
header .wrap, main {{ max-width:1080px; margin:0 auto; }}
h1 {{ margin:0 0 8px; font-size:2.2rem; }}
header p {{ margin:0; color:#d7eef5; max-width:780px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:14px; margin:-30px auto 30px; max-width:1080px; padding:0 24px; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px; box-shadow:0 8px 24px rgba(20,40,60,.07); }}
.kpi b {{ display:block; font-size:1.55rem; color:var(--accent); }}
.kpi span {{ color:var(--muted); font-size:.86rem; }}
main {{ padding:0 24px 60px; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:24px; margin:20px 0; }}
h2 {{ margin-top:0; font-size:1.35rem; }}
table {{ width:100%; border-collapse:collapse; font-size:.92rem; }}
th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }}
th {{ color:var(--muted); font-weight:600; }}
img {{ width:100%; height:auto; border-radius:10px; border:1px solid var(--line); margin-top:12px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
a {{ color:var(--accent); }}
code {{ background:#eef3f7; padding:2px 6px; border-radius:5px; }}
footer {{ color:var(--muted); text-align:center; padding:30px 24px 50px; font-size:.86rem; }}
</style>
</head>
<body>
<header><div class="wrap">
<h1>Predictive Maintenance Pipeline</h1>
<p>Cost-sensitive machine-failure prediction on the AI4I 2020 benchmark. Physics-inspired features, calibrated probabilities and a maintenance policy that trades missed failures against unnecessary inspections.</p>
</div></header>
<div class="kpis">
<div class="kpi"><b>{metrics['roc_auc']:.3f}</b><span>ROC-AUC</span></div>
<div class="kpi"><b>{metrics['pr_auc']:.3f}</b><span>PR-AUC</span></div>
<div class="kpi"><b>{metrics['recall']:.1%}</b><span>Failure recall</span></div>
<div class="kpi"><b>{metrics['precision']:.1%}</b><span>Inspection precision</span></div>
<div class="kpi"><b>{policy['cost_per_1000']:.0f}</b><span>Cost per 1,000</span></div>
</div>
<main>
<section>
<h2>What this project demonstrates</h2>
<p>Sensor data is cleaned, split and enriched with thermal, power and wear interactions. Four classifiers are compared with stratified cross-validation. The winner is calibrated, a cost-sensitive maintenance threshold is selected on validation, and the test set is evaluated once. Failure-mode coverage and product-type slices show where the policy works.</p>
<p><strong>Selected model:</strong> {model_name.replace('_', ' ')} - <strong>Threshold:</strong> {metrics['threshold']:.2f} - <strong>Cost ratio:</strong> {policy['fn_cost']:.0f}:{policy['fp_cost']:.0f}</p>
</section>
<section>
<h2>Model comparison</h2>
<table><thead><tr><th>Model</th><th>CV PR-AUC</th><th>CV ROC-AUC</th></tr></thead><tbody>{comparison_rows}</tbody></table>
<img src="figures/05_model_comparison.png" alt="Model comparison">
</section>
<section>
<h2>Test-set evaluation</h2>
<div class="grid">
<div><img src="figures/06_roc_pr.png" alt="ROC and precision-recall curves"></div>
<div><img src="figures/07_calibration.png" alt="Calibration curve"></div>
</div>
<img src="figures/08_confusion_matrix.png" alt="Confusion matrix">
<img src="figures/09_policy_curve.png" alt="Maintenance policy curve">
</section>
<section>
<h2>Risk tiers</h2>
<table><thead><tr><th>Tier</th><th>Machines</th><th>Failures</th><th>Observed failure rate</th></tr></thead><tbody>{tier_rows}</tbody></table>
<img src="figures/13_risk_tiers.png" alt="Risk tiers">
</section>
<section>
<h2>Failure-mode coverage</h2>
<table><thead><tr><th>Mode</th><th>Occurrences</th><th>Detected</th><th>Recall</th></tr></thead><tbody>{mode_rows}</tbody></table>
<img src="figures/11_failure_mode_recall.png" alt="Failure-mode recall">
</section>
<section>
<h2>Feature importance</h2>
<table><thead><tr><th>Feature</th><th>Mean drop in AP</th><th>Std</th></tr></thead><tbody>{importance_rows}</tbody></table>
<img src="figures/10_feature_importance.png" alt="Permutation importance">
</section>
<section>
<h2>Reproduce</h2>
<p><code>python scripts/fetch_data.py</code><br><code>python -m maintenance.pipeline</code><br><code>python -m maintenance.cli check</code><br><code>streamlit run dashboard/app.py</code></p>
<p>Full write-up: <a href="RESULTS.md">RESULTS.md</a> - <a href="METHODOLOGY.md">METHODOLOGY.md</a> - <a href="MODEL_CARD.md">MODEL_CARD.md</a></p>
</section>
</main>
<footer>Dataset: AI4I 2020 (CC BY 4.0). Code: MIT. Metrics come from a fixed test split; see the manifest for checksums.</footer>
</body>
</html>
"""
    path = paths["docs"] / "index.html"
    path.write_text(html, encoding="utf-8")

    results = f"""# Results

**Selected model:** {model_name.replace('_', ' ')} (isotonic calibration)
**Operating threshold:** {metrics['threshold']:.2f}

| Metric | Value |
| --- | ---: |
| ROC-AUC | {metrics['roc_auc']:.4f} |
| PR-AUC | {metrics['pr_auc']:.4f} |
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1 | {metrics['f1']:.4f} |
| Brier | {metrics['brier']:.4f} |
| Log loss | {metrics['log_loss']:.4f} |
| Flagged rate | {metrics['flagged_rate']:.4f} |

Confusion matrix: TN {metrics['confusion_matrix']['tn']:,} - FP {metrics['confusion_matrix']['fp']:,} - FN {metrics['confusion_matrix']['fn']:,} - TP {metrics['confusion_matrix']['tp']:,}.

## Model comparison

{_comparison_table(comparison)}

## Maintenance policy

- Threshold: {policy['threshold']:.2f}
- Cost ratio: {policy['fn_cost']:.0f}:{policy['fp_cost']:.0f}
- Expected cost per 1,000: {policy['cost_per_1000']:.0f}
- Recall: {policy['recall']:.4f}
- Precision: {policy['precision']:.4f}

## Failure-mode coverage

{_mode_table(modes)}

## Risk tiers

{_tier_table(tiers)}

## Top features

{_markdown_table(importance.head(12)[['feature', 'importance_mean', 'importance_std']])}
"""
    (paths["docs"] / "RESULTS.md").write_text(results, encoding="utf-8")
    (paths["docs"] / "MODEL_CARD.md").write_text(
        (paths["reports"] / "MODEL_CARD.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    feed = {
        "summary": summary,
        "model": model_name,
        "metrics": metrics,
        "comparison": comparison.to_dict(orient="records"),
        "policy": policy,
        "modes": modes.to_dict(orient="records"),
        "tiers": tiers.to_dict(orient="records"),
        "importance": importance.head(15).to_dict(orient="records"),
    }
    (paths["dashboard"] / "feed.json").write_text(json.dumps(feed, indent=2), encoding="utf-8")
    return path
