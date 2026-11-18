"""ACP : valeurs propres, coordonnees, contributions, cos2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pca_segmentation.features.scaling import Standardizer, drop_constant


@dataclass
class PCAResult:
    eigenvalues: pd.DataFrame  # valeur propre, % variance, % cumule
    ind_coords: pd.DataFrame  # coordonnees des individus
    ind_contrib: pd.DataFrame  # contributions des individus aux axes (%)
    ind_cos2: pd.DataFrame  # qualite de representation des individus
    var_coords: pd.DataFrame  # correlations variables-axes (cercle des correlations)
    var_contrib: pd.DataFrame  # contributions des variables aux axes (%)
    var_cos2: pd.DataFrame
    n_components: int
    scaler: Standardizer
    loadings: pd.DataFrame  # vecteurs propres (variables x axes)

    def project(self, X_sup: pd.DataFrame) -> pd.DataFrame:
        """Coordonnees d'individus supplementaires (non utilises pour construire les axes)."""
        Z = self.scaler.transform(X_sup)
        return Z @ self.loadings


def choose_n_components(eigenvalues: np.ndarray, target_variance: float = 0.80) -> int:
    """Plus petit k qui atteint la variance cible, borne par le critere de Kaiser (lambda > 1).

    Kaiser seul retient trop d'axes quand il y a beaucoup de variables ; la variance
    cumulee seule peut en retenir de non interpretables. On prend le minimum des deux,
    avec au moins 2 axes pour disposer d'un plan factoriel.
    """
    ratio = eigenvalues / eigenvalues.sum()
    k_var = int(np.searchsorted(np.cumsum(ratio), target_variance) + 1)
    k_kaiser = int((eigenvalues > 1).sum())
    return max(2, min(k_var, k_kaiser))


def fit_pca(
    X: pd.DataFrame, n_components: int | None = None, target_variance: float = 0.80
) -> PCAResult:
    """ACP normee par decomposition en valeurs singulieres de la matrice centree-reduite."""
    X = drop_constant(X)
    scaler = Standardizer()
    Z = scaler.fit_transform(X)
    n = Z.shape[0]

    U, S, Vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    eig = S**2 / n  # somme = p en ACP normee
    # convention de signe : la variable qui pese le plus est positive sur chaque axe
    signs = np.sign(Vt[np.arange(len(S)), np.abs(Vt).argmax(axis=1)])
    Vt = Vt * signs[:, None]
    U = U * signs[None, :]

    k = n_components or choose_n_components(eig, target_variance)
    k = min(k, len(S))
    axes = [f"F{i + 1}" for i in range(k)]

    eigen_df = pd.DataFrame(
        {
            "eigenvalue": eig,
            "variance_pct": eig / eig.sum() * 100,
            "cumulative_pct": np.cumsum(eig) / eig.sum() * 100,
        },
        index=pd.Index([f"F{i + 1}" for i in range(len(eig))], name="axis"),
    )

    coords = U[:, :k] * S[:k]
    ind_coords = pd.DataFrame(coords, index=X.index, columns=axes)
    d2 = (Z.to_numpy() ** 2).sum(axis=1)
    ind_cos2 = pd.DataFrame(coords**2 / d2[:, None], index=X.index, columns=axes)
    ind_contrib = pd.DataFrame(coords**2 / (n * eig[:k]) * 100, index=X.index, columns=axes)

    loadings = pd.DataFrame(Vt[:k].T, index=X.columns, columns=axes)
    var_coords = loadings * np.sqrt(eig[:k])
    var_cos2 = var_coords**2
    var_contrib = loadings**2 * 100

    return PCAResult(
        eigenvalues=eigen_df,
        ind_coords=ind_coords,
        ind_contrib=ind_contrib,
        ind_cos2=ind_cos2,
        var_coords=var_coords,
        var_contrib=var_contrib,
        var_cos2=var_cos2,
        n_components=k,
        scaler=scaler,
        loadings=loadings,
    )
