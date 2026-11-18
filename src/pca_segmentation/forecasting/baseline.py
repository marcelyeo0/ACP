"""Baseline naive saisonniere."""

from __future__ import annotations

import pandas as pd


def _future_index(y: pd.Series, h: int) -> pd.DatetimeIndex:
    return pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")


def seasonal_naive(y: pd.Series, h: int, m: int = 12) -> pd.Series:
    """y_hat(t+i) = y(t+i-m) : on reconduit la meme periode de l'an dernier.

    Reference obligatoire : un modele qui ne bat pas cette regle n'apporte rien.
    """
    if len(y) < m:
        raise ValueError(f"au moins {m} observations requises, {len(y)} fournies")
    values = [y.iloc[len(y) - m + (i % m)] for i in range(h)]
    return pd.Series(values, index=_future_index(y, h), name="seasonal_naive")


def seasonal_naive_drift(y: pd.Series, h: int, m: int = 12) -> pd.Series:
    """Naive saisonniere corrigee de la croissance annuelle observee sur les 12 derniers mois."""
    base = seasonal_naive(y, h, m)
    if len(y) < 2 * m:
        return base.rename("seasonal_naive_drift")
    growth = y.iloc[-m:].sum() / y.iloc[-2 * m : -m].sum()
    return (base * growth).rename("seasonal_naive_drift")
