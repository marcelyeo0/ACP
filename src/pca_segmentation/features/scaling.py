"""Centrage-reduction prealable a l'ACP."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Standardizer:
    """Centrage-reduction a l'ecart-type de population (convention ACP normee)."""

    mean_: pd.Series | None = None
    std_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> Standardizer:
        self.mean_ = X.mean()
        self.std_ = X.std(ddof=0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer non ajuste")
        X = X[self.mean_.index]
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def drop_constant(X: pd.DataFrame, tol: float = 1e-12) -> pd.DataFrame:
    """Retire les variables de variance nulle (impossibles a reduire)."""
    return X.loc[:, X.std(ddof=0) > tol]
