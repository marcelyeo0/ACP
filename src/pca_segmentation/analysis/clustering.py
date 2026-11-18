"""CAH (Ward) et k-means, choix du nombre de classes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score


@dataclass
class ClusteringResult:
    labels: pd.Series  # typologie finale (apres consolidation k-means), 1..k
    labels_cah: pd.Series  # partition CAH avant consolidation
    linkage: np.ndarray
    n_clusters: int
    diagnostics: pd.DataFrame  # par k : saut de Ward, silhouette, Calinski-Harabasz
    moved: int  # individus reaffectes par la consolidation


def ward_linkage(coords: pd.DataFrame, method: str = "ward") -> np.ndarray:
    return linkage(coords.to_numpy(), method=method)


def k_diagnostics(coords: pd.DataFrame, Z: np.ndarray, max_k: int = 8) -> pd.DataFrame:
    """Indicateurs pour choisir k : saut de hauteur du dendrogramme et qualite de partition."""
    heights = Z[:, 2][::-1]  # hauteurs de fusion, de la derniere (k=1) a la premiere
    rows = []
    for k in range(2, min(max_k, len(coords) - 1) + 1):
        lab = fcluster(Z, k, criterion="maxclust")
        rows.append(
            {
                "k": k,
                # ecart entre la fusion k -> k-1 et la fusion k+1 -> k : grand saut = coupe nette
                "ward_jump": heights[k - 2] - heights[k - 1],
                "silhouette": silhouette_score(coords, lab),
                "calinski_harabasz": calinski_harabasz_score(coords, lab),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def choose_k(diag: pd.DataFrame, min_k: int = 2) -> int:
    """k au plus grand saut de Ward, parmi k >= min_k (coupe la plus nette du dendrogramme).

    `min_k` permet d'ecarter une partition triviale : ici k=2 ne fait que couper
    l'axe F1 (effet taille, grands vs petits magasins), deja visible sans classification.
    """
    jump = diag.loc[diag.index >= min_k, "ward_jump"]
    return int(jump.idxmax())


def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
    order = pd.Series(labels).value_counts().index
    mapping = {old: new for new, old in enumerate(order, start=1)}
    return np.array([mapping[x] for x in labels])


def cluster(
    coords: pd.DataFrame,
    n_clusters: int | None = None,
    method: str = "ward",
    max_k: int = 8,
    random_state: int = 42,
    min_k: int = 2,
) -> ClusteringResult:
    """CAH sur les coordonnees factorielles, puis consolidation k-means.

    La CAH donne une partition stable et un dendrogramme lisible ; k-means initialise
    sur ses centres de classes corrige les affectations sous-optimales que la CAH,
    gloutonne, ne revisite jamais.
    """
    Z = ward_linkage(coords, method)
    diag = k_diagnostics(coords, Z, max_k)
    k = n_clusters or choose_k(diag, min_k)

    lab_cah = fcluster(Z, k, criterion="maxclust")
    centers = np.vstack([coords.to_numpy()[lab_cah == c].mean(axis=0) for c in np.unique(lab_cah)])
    km = KMeans(n_clusters=k, init=centers, n_init=1, random_state=random_state)
    lab_km = km.fit_predict(coords.to_numpy()) + 1

    lab_cah = _relabel_by_size(lab_cah)
    lab_km = _relabel_by_size(lab_km)
    # individus qui ont quitte la classe CAH majoritairement associee a leur classe finale
    moved = int(len(coords) - pd.crosstab(lab_cah, lab_km).to_numpy().max(axis=1).sum())

    idx = coords.index
    return ClusteringResult(
        labels=pd.Series(lab_km, index=idx, name="typology"),
        labels_cah=pd.Series(lab_cah, index=idx, name="typology_cah"),
        linkage=Z,
        n_clusters=k,
        diagnostics=diag,
        moved=moved,
    )
