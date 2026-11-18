"""Page Streamlit : ecarts d'assortiment."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pca_segmentation.config import get_settings

s = get_settings()
ex = s.exports
st.title("Assortiment theorique vs distribution effective")
st.caption(
    f"Gamme coeur d'une typologie = articles vendus par au moins "
    f"{s.core_range_threshold:.0%} de ses magasins sur la fenetre {s.reference_start} -> "
    f"{s.reference_end}. Potentiel = volume median des magasins de la typologie qui vendent l'article."
)

gaps = pd.read_csv(ex / "fact_store_gaps.csv")
fam = pd.read_csv(ex / "fact_family_gaps.csv")
missing = pd.read_csv(ex / "fact_missing_items.csv")

c1, c2, c3 = st.columns(3)
c1.metric("Couverture moyenne de la gamme coeur", f"{gaps['coverage_pct'].mean():.1f} %")
c2.metric(
    "References coeur manquantes (total)", f"{int(gaps['missing_items'].sum()):,}".replace(",", " ")
)
c3.metric(
    "Potentiel sur la gamme coeur",
    f"+{gaps['potential_units_missing'].sum() / gaps['actual_units_core'].sum() * 100:.2f} %",
)

st.subheader("Magasins les plus eloignes de leur gamme coeur")
cols = [
    "store_nbr",
    "city",
    "store_type",
    "typology",
    "core_items",
    "coverage_pct",
    "missing_items",
    "potential_units_missing",
    "potential_uplift_pct",
    "off_range_items",
]
st.dataframe(gaps.sort_values("coverage_pct")[cols].round(1), hide_index=True, width="stretch")

st.subheader("Couverture par typologie x famille")
k = st.selectbox("Typologie", sorted(fam["typology"].unique()), format_func=lambda x: f"T{x}")
st.bar_chart(fam[fam["typology"] == k].set_index("family")["coverage_pct"].sort_values())

st.subheader("Detail des references manquantes")
store = st.selectbox("Magasin", gaps.sort_values("coverage_pct")["store_nbr"])
st.dataframe(
    missing[missing["store_nbr"] == store].sort_values("potential_units", ascending=False).round(1),
    hide_index=True,
    width="stretch",
)
