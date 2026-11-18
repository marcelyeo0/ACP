"""Prevision a 3 mois par typologie de point de vente."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from pca_segmentation.forecasting.baseline import seasonal_naive


def holt_winters(y: pd.Series, h: int, m: int = 12) -> pd.Series:
    """Lissage exponentiel de Holt-Winters : tendance amortie, saisonnalite multiplicative.

    - multiplicative : le pic de decembre croit avec le niveau des ventes ;
    - tendance amortie : sur 3 mois, extrapoler une tendance lineaire sur-reagit.
    Sous deux saisons completes, on retombe sur la baseline.
    """
    y = y.asfreq("MS")
    if len(y) < 2 * m or (y <= 0).any():
        return seasonal_naive(y, h, m).rename("holt_winters")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(
            y,
            trend="add",
            damped_trend=True,
            seasonal="mul",
            seasonal_periods=m,
            initialization_method="estimated",
        ).fit(optimized=True)
    return fit.forecast(h).rename("holt_winters")


def forecast_with_intervals(
    y: pd.Series, h: int, residuals: pd.DataFrame | None = None, m: int = 12, level: float = 0.8
) -> pd.DataFrame:
    """Prevision ponctuelle + intervalle empirique tire des erreurs relatives du backtest.

    Les intervalles parametriques de Holt-Winters supposent des erreurs gaussiennes
    independantes ; les erreurs reellement observees en backtest, par pas d'horizon,
    sont plus honnetes.
    """
    point = holt_winters(y, h, m)
    out = pd.DataFrame({"forecast": point, "baseline": seasonal_naive(y, h, m).to_numpy()})
    if residuals is not None and not residuals.empty:
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
        rel = residuals.groupby("step")["rel_error"]
        lo = rel.quantile(lo_q).reindex(range(1, h + 1)).to_numpy()
        hi = rel.quantile(hi_q).reindex(range(1, h + 1)).to_numpy()
        # rel_error = (reel - prevu) / prevu  =>  reel = prevu * (1 + rel_error)
        out["lower"] = point.to_numpy() * (1 + np.nan_to_num(lo))
        out["upper"] = point.to_numpy() * (1 + np.nan_to_num(hi))
    out.index.name = "month"
    return out
