"""Cost-sensitive maintenance policy, failure-mode coverage and slices."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .config import (
    FAILURE_MODES,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    MODE_LABELS,
    TARGET,
)


def policy_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    fn_cost: float = FALSE_NEGATIVE_COST,
    fp_cost: float = FALSE_POSITIVE_COST,
) -> pd.DataFrame:
    """Expected maintenance cost per 1,000 machines at every threshold."""
    rows = []
    for threshold in np.round(np.arange(0.01, 1.0, 0.01), 2):
        predictions = (scores >= threshold).astype(int)
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        _, fp, fn, tp = matrix.ravel()
        total_cost = fn_cost * fn + fp_cost * fp
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                "flagged_rate": float(predictions.mean()),
                "missed_failures": int(fn),
                "false_alarms": int(fp),
                "total_cost": float(total_cost),
                "cost_per_1000": round(float(total_cost / len(labels) * 1000), 2),
            }
        )
    return pd.DataFrame(rows)


def choose_policy(curve: pd.DataFrame) -> dict:
    best = curve.loc[curve["cost_per_1000"].idxmin()]
    return {
        "threshold": float(best["threshold"]),
        "cost_per_1000": float(best["cost_per_1000"]),
        "recall": float(best["recall"]),
        "precision": float(best["precision"]),
        "flagged_rate": float(best["flagged_rate"]),
        "missed_failures": int(best["missed_failures"]),
        "false_alarms": int(best["false_alarms"]),
        "fn_cost": FALSE_NEGATIVE_COST,
        "fp_cost": FALSE_POSITIVE_COST,
    }


def failure_mode_recall(frame: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Share of each labelled failure mode caught at the operating threshold."""
    rows = []
    for mode in FAILURE_MODES:
        mask = frame[mode].to_numpy() == 1
        count = int(mask.sum())
        detected = int((predictions[mask] == 1).sum()) if count else 0
        rows.append(
            {
                "mode": mode,
                "label": MODE_LABELS[mode],
                "occurrences": count,
                "detected": detected,
                "recall": round(detected / count, 4) if count else np.nan,
            }
        )
    return pd.DataFrame(rows)


def operational_slices(scored: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for group, subset in scored.groupby(column, dropna=False):
        labels = subset[TARGET].to_numpy()
        predictions = subset["prediction"].to_numpy()
        tp = int(((predictions == 1) & (labels == 1)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())
        rows.append(
            {
                "slice": column,
                "group": str(group),
                "machines": len(subset),
                "failures": int(labels.sum()),
                "flag_rate": round(float(predictions.mean()), 4),
                "recall": round(tp / (tp + fn), 4) if tp + fn else np.nan,
                "precision": round(tp / (tp + fp), 4) if tp + fp else np.nan,
                "mean_score": round(float(subset["probability"].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def action_for_tier(tier: str) -> str:
    return {
        "Routine": "Continue normal operation",
        "Monitor": "Schedule condition monitoring",
        "Inspect": "Inspect before next shift",
        "Stop": "Stop and service immediately",
    }.get(tier, "Inspect before next shift")
