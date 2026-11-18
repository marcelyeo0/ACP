import numpy as np
import pandas as pd
import pytest

from pca_segmentation.analysis.clustering import choose_k, cluster
from pca_segmentation.analysis.interpretation import v_tests
from pca_segmentation.analysis.pca import choose_n_components, fit_pca


@pytest.fixture
def three_groups() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0, 0, 0, 0, 0], [6, 6, 0, 0, 0, 0], [0, 0, 6, 6, 0, 0]])
    X = np.vstack([c + rng.normal(size=(15, 6)) for c in centers])
    return pd.DataFrame(X, columns=[f"part_v{i}" for i in range(6)], index=range(1, 46))


def test_pca_invariants(three_groups):
    res = fit_pca(three_groups, n_components=3)
    p = three_groups.shape[1]
    # ACP normee : somme des valeurs propres = nombre de variables
    assert res.eigenvalues["eigenvalue"].sum() == pytest.approx(p)
    # contributions : 100 % par axe, pour les individus comme pour les variables
    assert np.allclose(res.ind_contrib.sum(), 100)
    assert np.allclose(res.var_contrib.sum(), 100)
    # cos2 dans [0, 1], et somme sur tous les axes = 1
    full = fit_pca(three_groups, n_components=p)
    assert np.allclose(full.ind_cos2.sum(axis=1), 1)
    # correlation variable-axe bornee par 1
    assert (res.var_coords.abs() <= 1 + 1e-9).all().all()


def test_projection_of_active_individuals_matches_coords(three_groups):
    res = fit_pca(three_groups, n_components=3)
    proj = res.project(three_groups)
    assert np.allclose(proj.to_numpy(), res.ind_coords.to_numpy())


def test_choose_n_components_kaiser_and_variance():
    eig = np.array([4.0, 1.5, 0.3, 0.2])
    assert choose_n_components(eig, 0.80) == 2
    # au moins 2 axes pour un plan factoriel
    assert choose_n_components(np.array([5.0, 0.5, 0.3, 0.2]), 0.5) == 2


def test_cluster_recovers_groups(three_groups):
    res = fit_pca(three_groups, n_components=3)
    cl = cluster(res.ind_coords)
    assert cl.n_clusters == 3
    truth = np.repeat([0, 1, 2], 15)
    assert pd.crosstab(cl.labels.to_numpy(), truth).max(axis=0).sum() == 45


def test_choose_k_largest_jump_above_min_k():
    diag = pd.DataFrame(
        {"ward_jump": [26.0, 9.4, 2.4, 1.5]}, index=pd.Index([2, 3, 4, 5], name="k")
    )
    assert choose_k(diag) == 2
    assert choose_k(diag, min_k=3) == 3


def test_v_tests_sign(three_groups):
    labels = pd.Series(np.repeat([1, 2, 3], 15), index=three_groups.index)
    vt = v_tests(three_groups, labels)
    assert vt.loc["part_v0", 2] > 1.96
    assert vt.loc["part_v0", 1] < -1.96
