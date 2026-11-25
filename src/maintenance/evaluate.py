"""Classification metrics, calibration and risk tiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import FALSE_NEGATIVE_COST, FALSE_POSITIVE_COST, RISK_TIERS, TARGET
from .features import model_input


def probabilities(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(model_input(frame))[:, 1]


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": round(float(threshold), 4),
        "rows": len(labels),
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
        "pr_auc": round(float(average_precision_score(labels, scores)), 4),
        "brier": round(float(brier_score_loss(labels, scores)), 4),
        "log_loss": round(float(log_loss(labels, np.clip(scores, 1e-6, 1 - 1e-6))), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "flagged_rate": round(float(predictions.mean()), 4),
        "missed_failures": int(fn),
        "false_alarms": int(fp),
        "cost": round(float(FALSE_NEGATIVE_COST * fn + FALSE_POSITIVE_COST * fp), 2),
    }


def threshold_sweep(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.01, 1.0, 0.01), 2):
        predictions = (scores >= threshold).astype(int)
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        _, fp, fn, tp = matrix.ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(labels, predictions, zero_division=0)),
                "recall": float(recall_score(labels, predictions, zero_division=0)),
                "f1": float(f1_score(labels, predictions, zero_division=0)),
                "flagged_rate": float(predictions.mean()),
                "missed_failures": int(fn),
                "false_alarms": int(fp),
                "true_positives": int(tp),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> pd.DataFrame:
    observed, predicted = calibration_curve(labels, scores, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"predicted": predicted, "observed": observed})


def curve_data(labels: np.ndarray, scores: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr": {"precision": precision.tolist(), "recall": recall.tolist()},
    }


def risk_tier(probability: float) -> str:
    for name, low, high in RISK_TIERS:
        if low <= probability < high:
            return name
    return "Stop"


def risk_tier_table(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    tiers = pd.cut(
        scores,
        bins=[tier[1] for tier in RISK_TIERS] + [1.01],
        labels=[tier[0] for tier in RISK_TIERS],
        include_lowest=True,
    ).astype(str)
    working = pd.DataFrame({"tier": tiers, TARGET: frame[TARGET].to_numpy()})
    grouped = working.groupby("tier", as_index=False).agg(
        machines=(TARGET, "count"), failures=(TARGET, "sum"), failure_rate=(TARGET, "mean")
    )
    order = {tier[0]: index for index, tier in enumerate(RISK_TIERS)}
    grouped["order"] = grouped["tier"].map(order)
    grouped = grouped.sort_values("order").drop(columns="order").reset_index(drop=True)
    grouped["failure_rate"] = grouped["failure_rate"].round(4)
    return grouped


def score_frame(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            TARGET: frame[TARGET].to_numpy(),
            "probability": np.round(scores, 5),
            "prediction": (scores >= threshold).astype(int),
            "risk_tier": [risk_tier(value) for value in scores],
            "type": frame["type"].to_numpy(),
            "tool_wear_band": frame["tool_wear_band"].to_numpy(),
            "tool_wear_min": frame["tool_wear_min"].to_numpy(),
            "torque_nm": frame["torque_nm"].to_numpy(),
        }
    )
    for mode in ["twf", "hdf", "pwf", "osf", "rnf"]:
        result[mode] = frame[mode].to_numpy()
    result["correct"] = result[TARGET] == result["prediction"]
    return result
