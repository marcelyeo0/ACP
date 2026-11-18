"""Page Streamlit : previsions a 3 mois."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pca_segmentation.config import get_settings

s = get_settings()
ex = s.exports
st.title("Prevision a 3 mois par typologie")
st.caption(
    "Unites vendues par magasin actif et par mois, a perimetre constant "
    "(articles vendus chaque mois de l'historique)."
)

hist = pd.read_csv(ex / "fact_typology_monthly.csv", parse_dates=["month"])
fc = pd.read_csv(ex / "fact_forecast.csv", parse_dates=["month"])

k = st.selectbox("Typologie", sorted(hist["typology"].unique()), format_func=lambda x: f"T{x}")
h = hist[hist["typology"] == k].set_index("month")["units_per_store"].rename("historique")
f = fc[fc["typology"] == k].set_index("month")
chart = pd.concat(
    [
        h,
        f["forecast"].rename("prevision"),
        f["baseline"].rename("naive saisonniere"),
        f.get("lower", pd.Series(dtype=float)).rename("borne basse 80 %"),
        f.get("upper", pd.Series(dtype=float)).rename("borne haute 80 %"),
    ],
    axis=1,
)
st.line_chart(chart)
st.dataframe(f.drop(columns="typology").round(0), width="stretch")

st.subheader("Backtest par origine glissante (12 origines x 3 mois)")
st.dataframe(pd.read_csv(ex / "backtest_summary.csv").round(2), hide_index=True)
st.markdown("**MAPE par typologie**")
st.dataframe(pd.read_csv(ex / "backtest_by_typology.csv").round(2), hide_index=True)

bt = pd.read_csv(ex / "fact_backtest.csv", parse_dates=["origin", "month"])
bt = bt[bt["typology"] == k]
bt["abs_pct_error"] = (bt["actual"] - bt["forecast"]).abs() / bt["actual"] * 100
st.markdown("**Erreur absolue en % par pas d'horizon**")
st.bar_chart(bt.groupby(["step", "model"])["abs_pct_error"].mean().unstack())
