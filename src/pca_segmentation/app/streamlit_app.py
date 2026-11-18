"""Entree du tableau de bord Streamlit."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from pca_segmentation.config import get_settings

st.set_page_config(page_title="Segmentation & prevision", layout="wide")
s = get_settings()
summary_path = s.exports / "summary.json"

st.title("Segmentation de points de vente par ACP & prevision de distribution")
st.caption(
    "Corporacion Favorita (Equateur) — 54 magasins, ~4 000 articles, ventes journalieres 2013-2017."
)

if not summary_path.exists():
    st.error("Aucun resultat : lancer `uv run python -m pca_segmentation.pipeline`.")
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
bt = summary["backtest"]

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Magasins analyses",
    summary["n_stores_active"],
    f"{len(summary['stores_supplementary'])} en supplementaire",
)
c2.metric(
    "Axes ACP retenus",
    summary["pca_n_components"],
    f"{summary['pca_cum_variance_pct']} % de variance",
    delta_color="off",
)
c3.metric(
    "Typologies",
    summary["n_clusters"],
    f"ARI vs format = {summary['ari_vs_store_type']}",
    delta_color="off",
)
c4.metric(
    "MAPE prevision 3 mois",
    f"{bt['holt_winters']['mape']:.1f} %",
    f"{-bt['holt_winters']['mape_gain_vs_baseline_pct']:.0f} % vs naive saisonniere",
    delta_color="inverse",
)

st.subheader("Demarche")
st.markdown(
    """
1. **Extraction SQL** (DuckDB sur parquet) : ventes magasin x famille x mois, presence article x magasin.
2. **Variables** : part de chaque famille dans les ventes (mix) et profondeur de gamme (% des references vendues).
3. **ACP normee** : axes retenus par Kaiser + 80 % de variance ; nommes via contributions et cos2.
4. **CAH Ward + consolidation k-means** sur les coordonnees factorielles.
5. **Prevision 3 mois** par typologie, a perimetre constant, backtest par origine glissante contre baseline.
6. **Ecarts** : gamme coeur de chaque typologie vs ce que chaque magasin vend reellement.
"""
)

st.subheader("Typologies")
dim = pd.read_csv(s.exports / "dim_stores.csv")
profiles = pd.read_csv(s.exports / "typology_profiles.csv")
st.dataframe(profiles, hide_index=True, width="stretch")
st.markdown("**Croisement avec le format magasin Favorita (A-E)**")
st.dataframe(pd.read_csv(s.exports / "typology_vs_store_type.csv"), hide_index=True)
st.caption("Pages detaillees dans la barre laterale : Segmentation, Forecast, Ecarts.")
