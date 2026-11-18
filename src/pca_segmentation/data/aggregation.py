"""Construction de la matrice point de vente x produit."""

from __future__ import annotations

import pandas as pd


def reference_window(monthly: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    m = pd.to_datetime(monthly["month"])
    return monthly[(m >= pd.Timestamp(start)) & (m <= pd.Timestamp(end))]


def store_family_matrix(monthly: pd.DataFrame, value: str = "units") -> pd.DataFrame:
    """Matrice magasin x famille (somme de `value` sur la periode fournie)."""
    return monthly.pivot_table(
        index="store_nbr", columns="family", values=value, aggfunc="sum", fill_value=0.0
    ).sort_index()


def presence_matrix(store_items: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'articles distincts vendus par magasin x famille."""
    return (
        store_items.groupby(["store_nbr", "family"])["item_nbr"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )


def typology_monthly(monthly: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Ventes mensuelles moyennes par magasin actif, par typologie.

    On divise par le nombre de magasins actifs du mois : les ouvertures de magasins
    ne doivent pas apparaitre comme de la croissance.
    """
    df = monthly.merge(labels.rename("typology"), left_on="store_nbr", right_index=True)
    per_store = df.groupby(["typology", "month", "store_nbr"], as_index=False)["units"].sum()
    per_store = per_store[per_store["units"] > 0]
    agg = per_store.groupby(["typology", "month"]).agg(
        units=("units", "sum"), n_stores=("store_nbr", "nunique")
    )
    agg["units_per_store"] = agg["units"] / agg["n_stores"]
    return agg.reset_index()
