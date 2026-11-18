"""Ecart entre assortiment theorique et distribution effective."""

from __future__ import annotations

import pandas as pd


def theoretical_assortment(
    store_items: pd.DataFrame, labels: pd.Series, threshold: float = 0.8
) -> pd.DataFrame:
    """Assortiment theorique (gamme coeur) d'une typologie.

    Un article en fait partie s'il est vendu par au moins `threshold` des magasins de la
    typologie. On en deduit le volume attendu d'un magasin qui le reference :
    la mediane des magasins de la typologie qui le vendent.
    """
    df = store_items.merge(labels.rename("typology"), left_on="store_nbr", right_index=True)
    n_stores = labels.value_counts().rename("n_stores")
    stats = (
        df.groupby(["typology", "item_nbr", "family"])
        .agg(n_selling=("store_nbr", "nunique"), median_units=("units", "median"))
        .reset_index()
        .merge(n_stores, left_on="typology", right_index=True)
    )
    stats["presence_rate"] = stats["n_selling"] / stats["n_stores"]
    core = stats[stats["presence_rate"] >= threshold]
    cols = ["typology", "item_nbr", "family", "presence_rate", "median_units"]
    return core[cols].reset_index(drop=True)


def store_gaps(
    store_items: pd.DataFrame, labels: pd.Series, core: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare chaque magasin a la gamme coeur de sa typologie.

    Retourne :
    - le detail des articles manquants (gamme coeur non vendue) avec volume potentiel ;
    - une synthese par magasin : taux de couverture (distribution numerique de la gamme
      coeur), nombre de manquants, volume potentiel manquant, references hors gamme.
    """
    labels = labels.rename_axis("store_nbr")
    sold = store_items[["store_nbr", "item_nbr", "units"]]
    expected = labels.rename("typology").reset_index().merge(core, on="typology")
    merged = expected.merge(sold, on=["store_nbr", "item_nbr"], how="left")
    merged["present"] = merged["units"].notna()

    missing = merged.loc[
        ~merged["present"], ["store_nbr", "typology", "item_nbr", "family", "median_units"]
    ].rename(columns={"median_units": "potential_units"})

    per_store = (
        merged.groupby(["store_nbr", "typology"])
        .agg(
            core_items=("item_nbr", "size"),
            core_present=("present", "sum"),
            actual_units_core=("units", "sum"),
        )
        .reset_index()
    )
    per_store["coverage_pct"] = per_store["core_present"] / per_store["core_items"] * 100
    per_store["missing_items"] = per_store["core_items"] - per_store["core_present"]
    pot = missing.groupby("store_nbr")["potential_units"].sum()
    per_store["potential_units_missing"] = per_store["store_nbr"].map(pot).fillna(0.0)
    per_store["potential_uplift_pct"] = (
        per_store["potential_units_missing"] / per_store["actual_units_core"] * 100
    )

    core_keys = pd.MultiIndex.from_frame(core[["typology", "item_nbr"]])
    off = sold.assign(typology=sold["store_nbr"].map(labels)).dropna(subset=["typology"])
    off_keys = pd.MultiIndex.from_arrays([off["typology"].astype(int), off["item_nbr"]])
    off_range = off[~off_keys.isin(core_keys)].groupby("store_nbr").size()
    per_store["off_range_items"] = per_store["store_nbr"].map(off_range).fillna(0).astype(int)
    return missing.reset_index(drop=True), per_store


def family_gaps(missing: pd.DataFrame, core: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Taux de couverture de la gamme coeur par typologie x famille."""
    n_stores = labels.value_counts()
    expected = core.groupby(["typology", "family"]).size().rename("core_items").reset_index()
    expected["expected_slots"] = expected["core_items"] * expected["typology"].map(n_stores)
    miss = (
        missing.groupby(["typology", "family"])
        .agg(missing_slots=("item_nbr", "size"), potential_units=("potential_units", "sum"))
        .reset_index()
    )
    out = expected.merge(miss, on=["typology", "family"], how="left").fillna(
        {"missing_slots": 0, "potential_units": 0.0}
    )
    out["coverage_pct"] = (1 - out["missing_slots"] / out["expected_slots"]) * 100
    return out.sort_values(["typology", "coverage_pct"]).reset_index(drop=True)
