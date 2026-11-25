"""End-to-end orchestration for the predictive-maintenance pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import (
    ARTIFACTS,
    CV_FOLDS,
    DASHBOARD_DIR,
    DOCS_DIR,
    DOCS_FIGURES_DIR,
    FIGURES_DIR,
    MANIFEST_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    REPORT_FIGURES_DIR,
    REPORTS_DIR,
    ROOT,
    TARGET,
)
from .data import dataset_summary, load_raw, median_template, split_data
from .evaluate import (
    calibration_table,
    curve_data,
    metrics_at_threshold,
    probabilities,
    risk_tier_table,
    score_frame,
)
from .explain import permutation_importance_table
from .manifest import build_manifest, write_manifest
from .model import calibrate_model, save_model, select_model
from .plots import make_all
from .policy import choose_policy, failure_mode_recall, operational_slices, policy_curve
from .report import write_model_card, write_report, write_site


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _split_summary(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> dict:
    return {
        "train_rows": len(train),
        "train_defaults": int(train[TARGET].sum()),
        "train_rate": round(float(train[TARGET].mean()), 4),
        "val_rows": len(validation),
        "val_defaults": int(validation[TARGET].sum()),
        "val_rate": round(float(validation[TARGET].mean()), 4),
        "test_rows": len(test),
        "test_defaults": int(test[TARGET].sum()),
        "test_rate": round(float(test[TARGET].mean()), 4),
    }


def _prepare_directories() -> dict[str, Path]:
    paths = {
        "root": ROOT,
        "artifacts": ARTIFACTS,
        "figures": FIGURES_DIR,
        "dashboard": DASHBOARD_DIR,
        "reports": REPORTS_DIR,
        "report_figures": REPORT_FIGURES_DIR,
        "docs": DOCS_DIR,
        "docs_figures": DOCS_FIGURES_DIR,
        "model": MODEL_PATH,
        "manifest": MANIFEST_PATH,
    }
    for key in ["artifacts", "figures", "dashboard", "reports", "report_figures", "docs", "docs_figures"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def run_pipeline(
    frame: pd.DataFrame | None = None,
    *,
    folds: int = CV_FOLDS,
    seed: int = RANDOM_STATE,
    make_plots: bool = True,
) -> dict:
    paths = _prepare_directories()
    data = load_raw() if frame is None else frame.copy()
    if TARGET not in data.columns:
        raise ValueError(f"input frame needs a {TARGET!r} column")

    summary = dataset_summary(data)
    train, validation, test = split_data(data, seed=seed)
    split_summary = _split_summary(train, validation, test)

    model_name, fitted, comparison = select_model(train, folds=folds, seed=seed)
    calibrated = calibrate_model(fitted, train, folds=min(3, folds))

    validation_scores = probabilities(calibrated, validation)
    validation_curve = policy_curve(validation[TARGET].to_numpy(), validation_scores)
    policy = choose_policy(validation_curve)

    test_scores = probabilities(calibrated, test)
    metrics = metrics_at_threshold(test[TARGET].to_numpy(), test_scores, policy["threshold"])
    metrics["model"] = model_name
    metrics["calibration"] = "isotonic"
    test_curve = policy_curve(test[TARGET].to_numpy(), test_scores)
    test_optimal = choose_policy(test_curve)

    calibration = calibration_table(test[TARGET].to_numpy(), test_scores)
    curves = curve_data(test[TARGET].to_numpy(), test_scores)
    tiers = risk_tier_table(test, test_scores)
    scored = score_frame(test, test_scores, policy["threshold"])
    predictions = scored["prediction"].to_numpy()
    modes = failure_mode_recall(test, predictions)
    slices = pd.concat(
        [operational_slices(scored, "type"), operational_slices(scored, "tool_wear_band")],
        ignore_index=True,
    )
    importance = permutation_importance_table(calibrated, test, repeats=5, seed=seed)

    save_model(calibrated, paths["model"])
    comparison.to_csv(paths["artifacts"] / "model_comparison.csv", index=False)
    test_curve.to_csv(paths["artifacts"] / "policy_curve.csv", index=False)
    calibration.to_csv(paths["artifacts"] / "calibration.csv", index=False)
    modes.to_csv(paths["artifacts"] / "failure_modes.csv", index=False)
    slices.to_csv(paths["artifacts"] / "operational_slices.csv", index=False)
    tiers.to_csv(paths["artifacts"] / "risk_tiers.csv", index=False)
    importance.to_csv(paths["artifacts"] / "feature_importance.csv", index=False)
    scored.to_csv(paths["artifacts"] / "test_scores.csv", index=False)
    (paths["dashboard"] / "input_template.json").write_text(
        json.dumps(_jsonable(median_template(data)), indent=2), encoding="utf-8"
    )

    metrics_payload = {
        "project": "predictive-maintenance-pipeline",
        "version": __version__,
        "dataset": "AI4I 2020 Predictive Maintenance",
        "summary": summary,
        "splits": split_summary,
        "model": model_name,
        "comparison": comparison.to_dict(orient="records"),
        "policy": policy,
        "test_optimal": test_optimal,
        "test_metrics": metrics,
        "failure_modes": modes.to_dict(orient="records"),
        "operational_slices": slices.to_dict(orient="records"),
        "risk_tiers": tiers.to_dict(orient="records"),
        "top_features": importance.head(12).to_dict(orient="records"),
    }
    metrics_path = paths["artifacts"] / "metrics.json"
    metrics_path.write_text(json.dumps(_jsonable(metrics_payload), indent=2), encoding="utf-8")

    if make_plots:
        figure_paths = make_all(
            data,
            comparison,
            curves,
            calibration,
            metrics,
            test_curve,
            policy,
            importance,
            modes,
            slices,
            tiers,
            paths["figures"],
        )
        for figure in figure_paths.values():
            shutil.copy2(figure, paths["docs_figures"] / figure.name)
            shutil.copy2(figure, paths["report_figures"] / figure.name)
        figure_paths = list(figure_paths.values())
    else:
        figure_paths = []

    write_report(
        summary=summary,
        split_summary=split_summary,
        model_name=model_name,
        comparison=comparison,
        policy=policy,
        test_optimal=test_optimal,
        metrics=metrics,
        modes=modes,
        slices=slices,
        tiers=tiers,
        importance=importance,
        paths=paths,
    )
    write_model_card(
        summary=summary,
        model_name=model_name,
        metrics=metrics,
        policy=policy,
        modes=modes,
        paths=paths,
    )
    write_site(
        summary=summary,
        model_name=model_name,
        metrics=metrics,
        comparison=comparison,
        policy=policy,
        modes=modes,
        tiers=tiers,
        importance=importance,
        paths=paths,
    )

    manifest_paths = [
        paths["model"],
        metrics_path,
        paths["artifacts"] / "model_comparison.csv",
        paths["artifacts"] / "policy_curve.csv",
        paths["artifacts"] / "calibration.csv",
        paths["artifacts"] / "failure_modes.csv",
        paths["artifacts"] / "operational_slices.csv",
        paths["artifacts"] / "risk_tiers.csv",
        paths["artifacts"] / "feature_importance.csv",
        paths["artifacts"] / "test_scores.csv",
        paths["dashboard"] / "input_template.json",
        paths["dashboard"] / "feed.json",
        paths["reports"] / "REPORT.md",
        paths["reports"] / "MODEL_CARD.md",
        paths["docs"] / "index.html",
        paths["docs"] / "RESULTS.md",
        paths["docs"] / "MODEL_CARD.md",
        *figure_paths,
    ]
    manifest = build_manifest(
        manifest_paths,
        ROOT,
        metadata={
            "project": "predictive-maintenance-pipeline",
            "version": __version__,
            "model": model_name,
            "test_roc_auc": metrics["roc_auc"],
            "test_pr_auc": metrics["pr_auc"],
            "rows": summary["rows"],
        },
    )
    write_manifest(manifest, paths["manifest"])

    return {
        "model": model_name,
        "policy": policy,
        "metrics": metrics,
        "summary": summary,
        "splits": split_summary,
        "artifacts": str(paths["artifacts"]),
        "manifest": str(paths["manifest"]),
    }


def main() -> None:
    result = run_pipeline()
    metrics = result["metrics"]
    print(f"model={result['model']}")
    print(f"threshold={metrics['threshold']}")
    print(f"roc_auc={metrics['roc_auc']} pr_auc={metrics['pr_auc']} recall={metrics['recall']}")
    print(f"artifacts={result['artifacts']}")


if __name__ == "__main__":
    main()
