"""Parts de marche, indices de presence, variables d'assortiment."""

from __future__ import annotations

import pandas as pd

from pca_segmentation.data.aggregation import presence_matrix, store_family_matrix


def kept_families(items: pd.DataFrame, min_items: int) -> list[str]:
    """Familles avec assez de references pour qu'une part ou une presence ait un sens."""
    counts = items.groupby("family")["item_nbr"].nunique()
    return sorted(counts[counts >= min_items].index)


def family_shares(monthly_ref: pd.DataFrame, families: list[str]) -> pd.DataFrame:
    """Part de chaque famille dans les volumes du magasin (mix), en %.

    Les parts neutralisent l'effet taille : un hypermarche et une superette
    deviennent comparables sur la structure de leurs ventes.
    """
    mat = store_family_matrix(monthly_ref).reindex(columns=families, fill_value=0.0)
    return mat.div(mat.sum(axis=1), axis=0) * 100


def presence_index(store_items: pd.DataFrame, families: list[str]) -> pd.DataFrame:
    """Part des references de la famille (vendues dans le reseau) vendues par le magasin, en %.

    Mesure la profondeur d'assortiment : un magasin peut vendre beaucoup de boissons
    sur 20 references ou un peu sur 400.
    """
    pres = presence_matrix(store_items).reindex(columns=families, fill_value=0)
    network = (
        store_items.groupby("family")["item_nbr"].nunique().reindex(families).replace(0, pd.NA)
    )
    return (pres / network * 100).astype(float).fillna(0.0)


def build_features(
    monthly_ref: pd.DataFrame,
    store_items: pd.DataFrame,
    items: pd.DataFrame,
    stores: pd.Index,
    min_items: int = 5,
) -> pd.DataFrame:
    """Matrice individus (magasins) x variables actives : `part_<famille>` et `pres_<famille>`."""
    fams = kept_families(items, min_items)
    m = monthly_ref[monthly_ref["store_nbr"].isin(stores)]
    si = store_items[store_items["store_nbr"].isin(stores)]
    shares = family_shares(m, fams).add_prefix("part_")
    presence = presence_index(si, fams).add_prefix("pres_")
    X = shares.join(presence, how="inner").reindex(stores).fillna(0.0)
    X.index.name = "store_nbr"
    return X
