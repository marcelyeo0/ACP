"""Cercle des correlations, dendrogramme, plan factoriel."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

PALETTE = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f", "#9c6644", "#6d597a", "#b56576"]


def scree_plot(eigenvalues: pd.DataFrame, n_kept: int, max_axes: int = 15) -> plt.Figure:
    ev = eigenvalues.head(max_axes)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [PALETTE[0] if i < n_kept else "#c9c9c9" for i in range(len(ev))]
    ax.bar(ev.index, ev["variance_pct"], color=colors)
    ax2 = ax.twinx()
    ax2.plot(ev.index, ev["cumulative_pct"], color=PALETTE[1], marker="o")
    ax2.axhline(80, color=PALETTE[1], ls=":", lw=1)
    ax2.set_ylim(0, 105)
    ax.set_ylabel("% de variance")
    ax2.set_ylabel("% cumule")
    kaiser = (ev["eigenvalue"] > 1).sum()
    ax.set_title(f"Eboulis des valeurs propres — {n_kept} axes retenus (Kaiser : {kaiser})")
    fig.tight_layout()
    return fig


def correlation_circle(
    var_coords: pd.DataFrame,
    eigenvalues: pd.DataFrame,
    axes: tuple[str, str] = ("F1", "F2"),
    min_cos2: float = 0.3,
) -> plt.Figure:
    """Cercle des correlations : seules les variables bien representees (cos2 > seuil) sont nommees."""
    a, b = axes
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#888"))
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.axvline(0, color="#bbb", lw=0.8)
    for var, row in var_coords.iterrows():
        x, y = row[a], row[b]
        good = x**2 + y**2 >= min_cos2
        color = PALETTE[0] if var.startswith("part_") else PALETTE[1]
        ax.arrow(0, 0, x, y, color=color if good else "#ddd", alpha=0.9, head_width=0.015)
        if good:
            ax.text(x * 1.07, y * 1.07, var.lower(), fontsize=7, ha="center", color=color)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{a} ({eigenvalues.loc[a, 'variance_pct']:.1f} %)")
    ax.set_ylabel(f"{b} ({eigenvalues.loc[b, 'variance_pct']:.1f} %)")
    ax.set_title("Cercle des correlations (bleu : parts, orange : profondeur)")
    fig.tight_layout()
    return fig


def factor_map(
    coords: pd.DataFrame,
    labels: pd.Series,
    eigenvalues: pd.DataFrame,
    axes: tuple[str, str] = ("F1", "F2"),
    supplementary: pd.DataFrame | None = None,
) -> plt.Figure:
    a, b = axes
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, k in enumerate(sorted(labels.unique())):
        idx = labels[labels == k].index
        ax.scatter(coords.loc[idx, a], coords.loc[idx, b], s=50, color=PALETTE[i], label=f"T{k}")
        for s in idx:
            ax.annotate(str(s), (coords.loc[s, a], coords.loc[s, b]), fontsize=7, alpha=0.7)
    if supplementary is not None and not supplementary.empty:
        ax.scatter(supplementary[a], supplementary[b], marker="x", color="black", label="suppl.")
        for s, row in supplementary.iterrows():
            ax.annotate(str(s), (row[a], row[b]), fontsize=7)
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.axvline(0, color="#bbb", lw=0.8)
    ax.set_xlabel(f"{a} ({eigenvalues.loc[a, 'variance_pct']:.1f} %)")
    ax.set_ylabel(f"{b} ({eigenvalues.loc[b, 'variance_pct']:.1f} %)")
    ax.legend(title="Typologie")
    ax.set_title("Plan factoriel des points de vente")
    fig.tight_layout()
    return fig


def dendrogram_plot(Z: np.ndarray, labels: list, n_clusters: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    cut = (Z[-n_clusters, 2] + Z[-n_clusters + 1, 2]) / 2
    dendrogram(Z, labels=labels, color_threshold=cut, ax=ax, leaf_font_size=7)
    ax.axhline(cut, color=PALETTE[1], ls="--", lw=1)
    ax.set_title(f"Dendrogramme (Ward) — coupe a {n_clusters} classes")
    ax.set_ylabel("Hauteur de fusion (critere de Ward)")
    fig.tight_layout()
    return fig


def forecast_plot(history: pd.DataFrame, forecasts: pd.DataFrame, last_n: int = 30) -> plt.Figure:
    """Historique et prevision a 3 mois par typologie, avec intervalle empirique."""
    typos = sorted(history["typology"].unique())
    fig, axes = plt.subplots(len(typos), 1, figsize=(10, 2.6 * len(typos)), sharex=True)
    axes = np.atleast_1d(axes)
    for i, (ax, k) in enumerate(zip(axes, typos, strict=True)):
        h = history[history["typology"] == k].set_index("month")["units_per_store"].tail(last_n)
        f = forecasts[forecasts["typology"] == k].set_index("month")
        ax.plot(h.index, h.values, color=PALETTE[i], label="historique")
        ax.plot(f.index, f["forecast"], color=PALETTE[i], ls="--", marker="o", label="prevision")
        ax.plot(f.index, f["baseline"], color="#999", ls=":", marker=".", label="naive saisonniere")
        if "lower" in f:
            ax.fill_between(f.index, f["lower"], f["upper"], color=PALETTE[i], alpha=0.15)
        ax.set_title(f"Typologie T{k} — unites / magasin / mois", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, path) -> None:
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
