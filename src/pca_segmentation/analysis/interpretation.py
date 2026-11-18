"""Nommage des axes factoriels et profils des typologies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def axis_drivers(
    var_coords: pd.DataFrame, var_contrib: pd.DataFrame, axis: str, top: int = 6
) -> pd.DataFrame:
    """Variables qui construisent un axe : contribution > moyenne (100/p), triees.

    Nommer un axe sans regarder les contributions revient a lire des formes dans les nuages.
    """
    threshold = 100 / len(var_contrib)
    df = pd.DataFrame({"correlation": var_coords[axis], "contribution_pct": var_contrib[axis]})
    df = df[df["contribution_pct"] > threshold].sort_values("contribution_pct", ascending=False)
    df["pole"] = np.where(df["correlation"] > 0, "+", "-")
    return df.head(top)


def _pretty(var: str) -> str:
    kind, fam = var.split("_", 1)
    return f"{'part' if kind == 'part' else 'profondeur'} {fam.lower()}"


def describe_axes(
    var_coords: pd.DataFrame, var_contrib: pd.DataFrame, top: int = 3
) -> pd.DataFrame:
    """Resume textuel de chaque axe : pole positif vs pole negatif."""
    rows = []
    for axis in var_coords.columns:
        d = axis_drivers(var_coords, var_contrib, axis, top=20)
        pos = [_pretty(v) for v in d[d["pole"] == "+"].index[:top]]
        neg = [_pretty(v) for v in d[d["pole"] == "-"].index[:top]]
        rows.append(
            {
                "axis": axis,
                "pole_positif": ", ".join(pos) or "-",
                "pole_negatif": ", ".join(neg) or "-",
            }
        )
    return pd.DataFrame(rows).set_index("axis")


def v_tests(X: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Valeur-test de chaque variable dans chaque classe (Lebart, Morineau, Piron).

    v = (moyenne_classe - moyenne) / sqrt(s2/n_k * (N - n_k)/(N - 1)).
    |v| > 1.96 : la classe s'ecarte significativement de la moyenne sur la variable.
    """
    N = len(X)
    mean = X.mean()
    var = X.var(ddof=0)
    out = {}
    for k, idx in labels.groupby(labels).groups.items():
        nk = len(idx)
        if nk == N:
            continue
        sd = np.sqrt(var / nk * (N - nk) / (N - 1))
        out[k] = (X.loc[idx].mean() - mean) / sd.replace(0, np.nan)
    return pd.DataFrame(out)


def typology_profiles(X: pd.DataFrame, labels: pd.Series, top: int = 4) -> pd.DataFrame:
    """Variables les plus sur- et sous-representees par typologie."""
    vt = v_tests(X, labels)
    rows = []
    for k in vt.columns:
        s = vt[k].dropna().sort_values()
        rows.append(
            {
                "typology": k,
                "n_stores": int((labels == k).sum()),
                "sur_representees": ", ".join(
                    f"{_pretty(v)} ({s[v]:+.1f})" for v in s.index[::-1][:top] if s[v] > 1.96
                ),
                "sous_representees": ", ".join(
                    f"{_pretty(v)} ({s[v]:+.1f})" for v in s.index[:top] if s[v] < -1.96
                ),
            }
        )
    return pd.DataFrame(rows).set_index("typology")


def compare_with_reference(labels: pd.Series, reference: pd.Series) -> tuple[pd.DataFrame, float]:
    """Croise la typologie obtenue avec une segmentation existante (ex. format magasin)."""
    ref = reference.reindex(labels.index)
    return pd.crosstab(labels, ref), float(adjusted_rand_score(labels, ref))
