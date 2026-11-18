"""Typage, doublons, valeurs manquantes, traitement des outliers."""

from __future__ import annotations

import pandas as pd


def clean_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Type les colonnes, supprime doublons et lignes sans cle, borne les volumes a 0."""
    out = df.copy()
    out["month"] = pd.to_datetime(out["month"]).dt.to_period("M").dt.to_timestamp()
    out["store_nbr"] = out["store_nbr"].astype(int)
    out["family"] = out["family"].astype(str)
    out = out.dropna(subset=["store_nbr", "family", "month"])
    out = out.drop_duplicates(subset=["store_nbr", "family", "month"], keep="last")
    for col in ("units", "units_promo", "n_items_sold"):
        if col in out:
            out[col] = out[col].fillna(0).clip(lower=0)
    return out.reset_index(drop=True)


def eligible_stores(stores: pd.DataFrame, min_active_months: int = 12) -> pd.Index:
    """Magasins actifs sur toute la fenetre de reference.

    Un magasin ouvert en cours de fenetre (ex. n°52, avril 2017) aurait des parts
    calculees sur quelques mois seulement : il fausserait l'ACP. Il reste projetable
    en individu supplementaire.
    """
    mask = stores["active_months_ref"].fillna(0) >= min_active_months
    return pd.Index(stores.loc[mask, "store_nbr"].astype(int), name="store_nbr")


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Ecrete une serie a ses quantiles (outliers de saisie ou evenements exceptionnels)."""
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)


def flag_outliers_iqr(s: pd.Series, k: float = 3.0) -> pd.Series:
    """Booleen True pour les valeurs hors de [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)
