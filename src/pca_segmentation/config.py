"""Chemins, DSN, n_components, n_clusters, horizon de prevision."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Parametres du pipeline, surchargeables par variables d'environnement ou `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", extra="ignore", env_parse_none_str=""
    )

    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    outputs_dir: Path = Path("outputs")
    sql_dir: Path = Path("sql")

    # Fenetre de reference pour l'ACP et l'analyse d'ecart (12 mois glissants complets).
    reference_start: str = "2016-08-01"
    reference_end: str = "2017-07-31"
    # Fin de l'historique mensuel complet (aout 2017 est partiel dans la source).
    history_end: str = "2017-07-31"

    # Familles trop petites pour porter une information d'assortiment.
    min_items_per_family: int = 5

    pca_n_components: int | None = None  # None : Kaiser + variance cible
    pca_target_variance: float = 0.80
    cluster_method: str = "ward"
    n_clusters: int | None = None  # None : saut de Ward + silhouette
    max_clusters: int = 8
    # k=2 reproduit l'axe F1 (effet taille) : sans valeur operationnelle.
    min_clusters: int = 3

    forecast_horizon_months: int = 3
    backtest_n_origins: int = 12
    seasonal_period: int = 12

    # Seuil de presence au-dela duquel un article fait partie de l'assortiment theorique
    # d'une typologie (part des magasins de la typologie qui le vendent).
    core_range_threshold: float = 0.80

    random_state: int = 42

    def path(self, p: Path) -> Path:
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def raw(self) -> Path:
        return self.path(self.raw_data_dir)

    @property
    def processed(self) -> Path:
        return self.path(self.processed_data_dir)

    @property
    def outputs(self) -> Path:
        return self.path(self.outputs_dir)

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    @property
    def models(self) -> Path:
        return self.outputs / "models"

    @property
    def exports(self) -> Path:
        return self.outputs / "exports"

    @property
    def sql(self) -> Path:
        return self.path(self.sql_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
