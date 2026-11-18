"""Lecture des ventes depuis SQL et CSV."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from pca_segmentation.config import Settings, get_settings

HF_DATASET = "https://huggingface.co/datasets/hsnalmasri/favorita/resolve/main"
RAW_TABLES = ("train", "items", "stores", "transactions", "holidays_events")


def render_sql(name: str, settings: Settings | None = None) -> str:
    """Charge `sql/<name>.sql` et remplace les parametres `{...}`."""
    s = settings or get_settings()
    query = (s.sql / f"{name}.sql").read_text(encoding="utf-8-sig")
    return query.format(
        raw=s.raw.as_posix(),
        reference_start=s.reference_start,
        reference_end=s.reference_end,
        history_end=s.history_end,
    )


def run_sql(name: str, settings: Settings | None = None) -> pd.DataFrame:
    """Execute une requete du dossier `sql/` avec DuckDB sur les parquets bruts."""
    with duckdb.connect() as con:
        return con.sql(render_sql(name, settings)).df()


def check_raw_files(settings: Settings | None = None) -> list[Path]:
    """Liste les parquets manquants (voir `download_raw`)."""
    s = settings or get_settings()
    return [s.raw / f"{t}.parquet" for t in RAW_TABLES if not (s.raw / f"{t}.parquet").exists()]


def download_raw(settings: Settings | None = None) -> None:
    """Telecharge les parquets Favorita (~900 Mo pour `train`) depuis Hugging Face."""
    import urllib.request

    s = settings or get_settings()
    s.raw.mkdir(parents=True, exist_ok=True)
    for path in check_raw_files(s):
        print(f"telechargement {path.name} ...")
        urllib.request.urlretrieve(f"{HF_DATASET}/{path.name}", path)


def read_processed(name: str, settings: Settings | None = None) -> pd.DataFrame:
    s = settings or get_settings()
    return pd.read_parquet(s.processed / f"{name}.parquet")


def write_processed(df: pd.DataFrame, name: str, settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    s.processed.mkdir(parents=True, exist_ok=True)
    path = s.processed / f"{name}.parquet"
    df.to_parquet(path, index=isinstance(df.index, pd.MultiIndex) or df.index.name is not None)
    return path


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)
