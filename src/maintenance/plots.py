"""Matplotlib figures for maintenance EDA, evaluation and policy."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TARGET

GOOD, BAD, INK, ACCENT = "#2A9D8F", "#D1495B", "#243447", "#1F6F8B"
AMBER = "#E9C46A"
GRID = {"alpha": 0.25, "linewidth": 0.6}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def failure_balance(frame: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = frame[TARGET].value_counts().sort_index()
    bars = ax.bar(["Healthy", "Failure"], counts.values, color=[GOOD, BAD], width=0.55)
    for bar, value in zip(bars, counts.values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom")
    ax.set_ylabel("Machine observations")
    ax.set_title("Failure imbalance")
    ax.grid(axis="y", **GRID)
    return _save(fig, path)


def failure_modes(frame: pd.DataFrame, path: Path) -> Path:
    modes = ["twf", "hdf", "pwf", "osf", "rnf"]
    labels = ["Tool wear", "Heat diss.", "Power", "Overstrain", "Random"]
    counts = [int(frame[mode].sum()) for mode in modes]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts, color=ACCENT)
    for bar, value in zip(bars, counts, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value}", ha="center", va="bottom")
    ax.set_ylabel("Labelled occurrences")
    ax.set_title("Failure-mode breakdown (multi-label)")
    ax.grid(axis="y", **GRID)
    return _save(fig, path)


def sensor_distributions(frame: pd.DataFrame, path: Path) -> Path:
    enriched = frame.copy()
    enriched["power_kw"] = enriched["torque_nm"] * 2 * np.pi * enriched["rpm"] / 60000
    enriched["temp_delta_k"] = enriched["process_temp_k"] - enriched["air_temp_k"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, column, title in (
        (axes[0], "temp_delta_k", "Temperature delta (K)"),
        (axes[1], "power_kw", "Mechanical power (kW)"),
        (axes[2], "tool_wear_min", "Tool wear (min)"),
    ):
        healthy = enriched.loc[enriched[TARGET] == 0, column]
        failed = enriched.loc[enriched[TARGET] == 1, column]
        bins = np.linspace(enriched[column].quantile(0.01), enriched[column].quantile(0.99), 35)
        ax.hist(healthy, bins=bins, alpha=0.65, label="Healthy", color=GOOD, density=True)
        ax.hist(failed, bins=bins, alpha=0.65, label="Failure", color=BAD, density=True)
        ax.set_title(title)
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", **GRID)
    return _save(fig, path)


def correlation_heatmap(frame: pd.DataFrame, path: Path) -> Path:
    columns = [
        "air_temp_k",
        "process_temp_k",
        "rpm",
        "torque_nm",
        "tool_wear_min",
        TARGET,
    ]
    matrix = frame[columns].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), [column.replace("_", " ") for column in columns], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(columns)), [column.replace("_", " ") for column in columns], fontsize=8)
    for row in range(len(columns)):
        for col in range(len(columns)):
            ax.text(col, row, f"{matrix.values[row, col]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Sensor correlation structure")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, path)


def model_comparison(comparison: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(comparison))
    width = 0.38
    ax.bar(x - width / 2, comparison["pr_auc_mean"], width, label="PR-AUC", color=ACCENT)
    ax.bar(x + width / 2, comparison["roc_auc_mean"], width, label="ROC-AUC", color=GOOD)
    ax.set_xticks(x, [name.replace("_", " ") for name in comparison["model"]], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Cross-validated score")
    ax.set_title("Model comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", **GRID)
    return _save(fig, path)


def roc_pr(curves: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(curves["roc"]["fpr"], curves["roc"]["tpr"], color=BAD, label="ROC")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[1].plot(curves["pr"]["recall"], curves["pr"]["precision"], color=ACCENT)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve")
    for ax in axes:
        ax.grid(**GRID)
    return _save(fig, path)


def calibration(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Perfect")
    ax.plot(table["predicted"], table["observed"], marker="o", color=ACCENT, label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed failure rate")
    ax.set_title("Reliability diagram")
    ax.legend(frameon=False)
    ax.grid(**GRID)
    return _save(fig, path)


def confusion(matrix: dict, path: Path) -> Path:
    values = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(values, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{values[row, col]:,}", ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1], ["Predicted healthy", "Predicted failure"])
    ax.set_yticks([0, 1], ["Actual healthy", "Actual failure"])
    ax.set_title("Confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, path)


def policy_curve(curve: pd.DataFrame, chosen: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(curve["threshold"], curve["cost_per_1000"], color=BAD, label="Expected cost per 1,000")
    ax.axvline(chosen["threshold"], color=INK, linestyle="--", linewidth=1, label=f"Chosen {chosen['threshold']:.2f}")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Cost per 1,000 machines")
    ax.set_title("Cost-sensitive maintenance policy")
    ax.legend(frameon=False)
    ax.grid(**GRID)
    return _save(fig, path)


def feature_importance(table: pd.DataFrame, path: Path, top: int = 14) -> Path:
    subset = table.head(top).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(subset["feature"], subset["importance_mean"], xerr=subset["importance_std"], color=ACCENT)
    ax.set_xlabel("Drop in average precision when shuffled")
    ax.set_title("Permutation importance (test set)")
    ax.grid(axis="x", **GRID)
    return _save(fig, path)


def failure_mode_recall(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(table["label"], table["recall"], color=ACCENT)
    for bar, value, count in zip(bars, table["recall"], table["occurrences"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0%}\n(n={count})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Recall at operating threshold")
    ax.set_title("Failure-mode coverage")
    ax.grid(axis="y", **GRID)
    return _save(fig, path)


def operational_slices(table: pd.DataFrame, path: Path) -> Path:
    subset = table[table["slice"] == "type"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(subset))
    width = 0.36
    ax.bar(x - width / 2, subset["recall"], width, label="Recall", color=GOOD)
    ax.bar(x + width / 2, subset["precision"], width, label="Precision", color=ACCENT)
    ax.set_xticks(x, subset["group"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Product type")
    ax.set_ylabel("Score")
    ax.set_title("Performance by product type")
    ax.legend(frameon=False)
    ax.grid(axis="y", **GRID)
    return _save(fig, path)


def risk_tiers(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(table["tier"], table["failure_rate"], color=[GOOD, AMBER, "#F4A261", BAD], width=0.6)
    for bar, value, count in zip(bars, table["failure_rate"], table["machines"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1%}\n(n={count})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Observed failure rate")
    ax.set_title("Risk tiers on the test set")
    ax.set_ylim(0, min(1.0, max(table["failure_rate"]) * 1.35 + 0.05))
    ax.grid(axis="y", **GRID)
    return _save(fig, path)


def make_all(
    frame: pd.DataFrame,
    comparison: pd.DataFrame,
    curves: dict,
    calibration_table: pd.DataFrame,
    metrics: dict,
    policy: pd.DataFrame,
    chosen: dict,
    importance: pd.DataFrame,
    modes: pd.DataFrame,
    slices: pd.DataFrame,
    tiers: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    return {
        "01_failure_balance": failure_balance(frame, out_dir / "01_failure_balance.png"),
        "02_failure_modes": failure_modes(frame, out_dir / "02_failure_modes.png"),
        "03_sensor_distributions": sensor_distributions(frame, out_dir / "03_sensor_distributions.png"),
        "04_correlation_heatmap": correlation_heatmap(frame, out_dir / "04_correlation_heatmap.png"),
        "05_model_comparison": model_comparison(comparison, out_dir / "05_model_comparison.png"),
        "06_roc_pr": roc_pr(curves, out_dir / "06_roc_pr.png"),
        "07_calibration": calibration(calibration_table, out_dir / "07_calibration.png"),
        "08_confusion_matrix": confusion(metrics["confusion_matrix"], out_dir / "08_confusion_matrix.png"),
        "09_policy_curve": policy_curve(policy, chosen, out_dir / "09_policy_curve.png"),
        "10_feature_importance": feature_importance(importance, out_dir / "10_feature_importance.png"),
        "11_failure_mode_recall": failure_mode_recall(modes, out_dir / "11_failure_mode_recall.png"),
        "12_operational_slices": operational_slices(slices, out_dir / "12_operational_slices.png"),
        "13_risk_tiers": risk_tiers(tiers, out_dir / "13_risk_tiers.png"),
    }
