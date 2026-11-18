"""Page Streamlit : plan factoriel et typologies."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pca_segmentation.config import get_settings

s = get_settings()
ex, figs = s.exports, s.figures
st.title("Segmentation : ACP & CAH")

eig = pd.read_csv(ex / "pca_eigenvalues.csv", index_col=0)
axes = pd.read_csv(ex / "pca_axes.csv", index_col=0)
dim = pd.read_csv(ex / "dim_stores.csv")

col1, col2 = st.columns(2)
col1.image(str(figs / "eboulis.png"))
col1.dataframe(eig.head(10).round(2), width="stretch")
col2.image(str(figs / "dendrogramme.png"))
col2.dataframe(pd.read_csv(ex / "cluster_k_diagnostics.csv").round(3), hide_index=True)

st.subheader("Lecture des axes (variables a contribution > moyenne)")
st.dataframe(axes, width="stretch")

st.subheader("Plan factoriel")
fcols = [c for c in dim.columns if c.startswith("F")]
a = st.selectbox("Axe horizontal", fcols, index=0)
b = st.selectbox("Axe vertical", fcols, index=1)
plot = dim.dropna(subset=["typology"]).assign(
    typology=lambda d: "T" + d["typology"].astype(int).astype(str)
)
st.scatter_chart(plot, x=a, y=b, color="typology")

c1, c2 = st.columns(2)
c1.image(str(figs / "cercle_F1F2.png"))
if (figs / "cercle_F1F3.png").exists():
    c2.image(str(figs / "cercle_F1F3.png"))

st.subheader("Qualite de representation des magasins")
ind = pd.read_csv(ex / "pca_individuals.csv", header=[0, 1], index_col=0)
st.dataframe(ind["cos2"].round(2), width="stretch")

st.subheader("Magasins")
st.dataframe(
    dim[
        ["store_nbr", "city", "state", "store_type", "favorita_cluster", "typology", *fcols[:3]]
    ].round(2),
    hide_index=True,
    width="stretch",
)
