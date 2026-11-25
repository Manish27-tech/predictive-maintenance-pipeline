"""Permutation importance for the fitted maintenance pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from .config import TARGET
from .features import model_input


def permutation_importance_table(model, frame: pd.DataFrame, repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    inputs = model_input(frame)
    labels = frame[TARGET].to_numpy()
    result = permutation_importance(
        model,
        inputs,
        labels,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": inputs.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    table["importance_mean"] = table["importance_mean"].round(5)
    table["importance_std"] = table["importance_std"].round(5)
    return table
