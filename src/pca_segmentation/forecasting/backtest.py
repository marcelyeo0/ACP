"""Validation temporelle : MAPE, RMSE, comparaison a la baseline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from pca_segmentation.forecasting.baseline import seasonal_naive, seasonal_naive_drift
from pca_segmentation.forecasting.model import holt_winters

Forecaster = Callable[[pd.Series, int], pd.Series]

DEFAULT_MODELS: dict[str, Forecaster] = {
    "seasonal_naive": seasonal_naive,
    "seasonal_naive_drift": seasonal_naive_drift,
    "holt_winters": holt_winters,
}


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rolling_origin(
    y: pd.Series,
    h: int = 3,
    n_origins: int = 12,
    models: dict[str, Forecaster] | None = None,
    min_train: int = 24,
) -> pd.DataFrame:
    """Backtest par origine glissante (fenetre d'entrainement croissante).

    A chaque origine t, chaque modele est entraine sur y[:t] seulement et prevoit
    t+1..t+h. Aucune information future ne fuit dans l'entrainement.
    Retourne une ligne par (modele, origine, pas d'horizon).
    """
    models = models or DEFAULT_MODELS
    y = y.asfreq("MS")
    last_origin = len(y) - h
    first_origin = max(last_origin - n_origins + 1, min_train)
    rows = []
    for t in range(first_origin, last_origin + 1):
        train, test = y.iloc[:t], y.iloc[t : t + h]
        for name, fn in models.items():
            pred = fn(train, h)
            for step, (month, actual) in enumerate(test.items(), start=1):
                p = float(pred.iloc[step - 1])
                rows.append(
                    {
                        "model": name,
                        "origin": train.index[-1],
                        "month": month,
                        "step": step,
                        "actual": float(actual),
                        "forecast": p,
                        "rel_error": (float(actual) - p) / p if p else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame, baseline: str = "seasonal_naive") -> pd.DataFrame:
    """MAPE / RMSE par modele et gain relatif vs baseline (positif = meilleur que la baseline)."""
    rows = {
        name: {"mape": mape(d["actual"], d["forecast"]), "rmse": rmse(d["actual"], d["forecast"])}
        for name, d in bt.groupby("model")
    }
    summary = pd.DataFrame(rows).T
    summary.index.name = "model"
    base = summary.loc[baseline]
    summary["mape_gain_vs_baseline_pct"] = (1 - summary["mape"] / base["mape"]) * 100
    summary["rmse_gain_vs_baseline_pct"] = (1 - summary["rmse"] / base["rmse"]) * 100
    return summary.sort_values("mape")
