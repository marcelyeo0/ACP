"""Pipeline de bout en bout : extraction SQL -> ACP -> CAH -> prevision -> ecarts -> exports.

uv run python -m pca_segmentation.pipeline            # tout
uv run python -m pca_segmentation.pipeline --skip-extract
"""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from pca_segmentation.analysis.clustering import cluster
from pca_segmentation.analysis.interpretation import (
    compare_with_reference,
    describe_axes,
    typology_profiles,
)
from pca_segmentation.analysis.pca import fit_pca
from pca_segmentation.config import Settings, get_settings
from pca_segmentation.data.aggregation import reference_window, typology_monthly
from pca_segmentation.data.cleaning import clean_monthly_sales, eligible_stores
from pca_segmentation.data.loaders import (
    check_raw_files,
    download_raw,
    read_processed,
    run_sql,
    write_processed,
)
from pca_segmentation.features.build_features import build_features
from pca_segmentation.forecasting.backtest import rolling_origin, summarize
from pca_segmentation.forecasting.model import forecast_with_intervals
from pca_segmentation.reporting import figures as fig
from pca_segmentation.reporting.gap_analysis import family_gaps, store_gaps, theoretical_assortment


def extract(s: Settings) -> None:
    if check_raw_files(s):
        download_raw(s)
    print("extraction SQL (DuckDB) ...")
    write_processed(run_sql("dim_stores", s), "stores", s)
    write_processed(clean_monthly_sales(run_sql("extract_sales", s)), "monthly_sales", s)
    write_processed(run_sql("extract_store_items", s), "store_items", s)
    lfl = run_sql("extract_sales_lfl", s)
    lfl["month"] = lfl["month"].astype("datetime64[ns]")
    write_processed(lfl, "monthly_sales_lfl", s)


def run(s: Settings, skip_extract: bool = False) -> dict:
    if not skip_extract:
        extract(s)
    for d in (s.figures, s.models, s.exports):
        d.mkdir(parents=True, exist_ok=True)

    stores = read_processed("stores", s)
    monthly = read_processed("monthly_sales", s)
    monthly_lfl = read_processed("monthly_sales_lfl", s)
    store_items = read_processed("store_items", s)
    items = pd.read_parquet(s.raw / "items.parquet")

    # --- Features ----------------------------------------------------------------
    active = eligible_stores(stores)
    excluded = sorted(set(stores["store_nbr"]) - set(active))
    monthly_ref = reference_window(monthly, s.reference_start, s.reference_end)
    X = build_features(monthly_ref, store_items, items, active, s.min_items_per_family)
    X_sup = build_features(
        monthly_ref,
        store_items,
        items,
        pd.Index(excluded, name="store_nbr"),
        s.min_items_per_family,
    )
    print(f"matrice ACP : {X.shape[0]} magasins x {X.shape[1]} variables ; exclus : {excluded}")

    # --- ACP ---------------------------------------------------------------------
    pca = fit_pca(X, s.pca_n_components, s.pca_target_variance)
    sup_coords = pca.project(X_sup) if len(X_sup) else pd.DataFrame()
    axes_desc = describe_axes(pca.var_coords, pca.var_contrib, pca.eigenvalues)
    print(
        f"ACP : {pca.n_components} axes, {pca.eigenvalues['cumulative_pct'].iloc[pca.n_components - 1]:.1f} % de variance"
    )

    # --- Classification ----------------------------------------------------------
    cl = cluster(
        pca.ind_coords,
        s.n_clusters,
        s.cluster_method,
        s.max_clusters,
        s.random_state,
        s.min_clusters,
    )
    labels = cl.labels
    profiles = typology_profiles(X, labels)
    ref_type = stores.set_index("store_nbr")["store_type"]
    xtab, ari = compare_with_reference(labels, ref_type)
    print(
        f"typologies : k={cl.n_clusters}, {cl.moved} magasin(s) reaffecte(s) par k-means, ARI vs format Favorita = {ari:.2f}"
    )

    # --- Prevision ---------------------------------------------------------------
    # perimetre constant : neutralise les variations de couverture catalogue de la source
    hist = typology_monthly(monthly_lfl, labels)
    backtests, forecasts = [], []
    for k, g in hist.groupby("typology"):
        y = g.set_index("month")["units_per_store"].asfreq("MS")
        bt = rolling_origin(y, s.forecast_horizon_months, s.backtest_n_origins)
        bt["typology"] = k
        backtests.append(bt)
        fc = forecast_with_intervals(
            y, s.forecast_horizon_months, bt[bt["model"] == "holt_winters"], s.seasonal_period
        )
        fc["typology"] = k
        forecasts.append(fc.reset_index())
    backtest = pd.concat(backtests, ignore_index=True)
    forecast = pd.concat(forecasts, ignore_index=True)
    bt_summary = summarize(backtest)
    bt_by_typo = (
        backtest.groupby("typology")
        .apply(lambda d: summarize(d)[["mape"]].T.reset_index(drop=True), include_groups=False)
        .reset_index(level=1, drop=True)
    )
    print("backtest :\n" + bt_summary.round(2).to_string())

    # --- Ecarts d'assortiment ----------------------------------------------------
    si = store_items[store_items["store_nbr"].isin(active)]
    core = theoretical_assortment(si, labels, s.core_range_threshold)
    missing, per_store = store_gaps(si, labels, core)
    fam_gaps = family_gaps(missing, core, labels)
    per_store = per_store.merge(stores, on="store_nbr", how="left")

    # --- Figures -----------------------------------------------------------------
    fig.save(fig.scree_plot(pca.eigenvalues, pca.n_components), s.figures / "eboulis.png")
    fig.save(fig.correlation_circle(pca.var_coords, pca.eigenvalues), s.figures / "cercle_F1F2.png")
    if pca.n_components >= 3:
        fig.save(
            fig.correlation_circle(pca.var_coords, pca.eigenvalues, ("F1", "F3")),
            s.figures / "cercle_F1F3.png",
        )
    fig.save(
        fig.factor_map(pca.ind_coords, labels, pca.eigenvalues, supplementary=sup_coords),
        s.figures / "plan_factoriel.png",
    )
    fig.save(
        fig.dendrogram_plot(cl.linkage, list(X.index), cl.n_clusters),
        s.figures / "dendrogramme.png",
    )
    fig.save(fig.forecast_plot(hist, forecast), s.figures / "prevision.png")

    # --- Exports (Streamlit + Power BI) --------------------------------------------
    ex = s.exports
    dim = stores.set_index("store_nbr").join(labels).join(pca.ind_coords)
    dim.to_csv(ex / "dim_stores.csv")
    pca.eigenvalues.to_csv(ex / "pca_eigenvalues.csv")
    pd.concat(
        {"correlation": pca.var_coords, "contribution": pca.var_contrib, "cos2": pca.var_cos2},
        axis=1,
    ).to_csv(ex / "pca_variables.csv")
    pd.concat(
        {"coord": pca.ind_coords, "contribution": pca.ind_contrib, "cos2": pca.ind_cos2}, axis=1
    ).to_csv(ex / "pca_individuals.csv")
    sup_coords.to_csv(ex / "pca_supplementary.csv")
    axes_desc.to_csv(ex / "pca_axes.csv")
    X.to_csv(ex / "features.csv")
    cl.diagnostics.to_csv(ex / "cluster_k_diagnostics.csv")
    profiles.to_csv(ex / "typology_profiles.csv")
    xtab.to_csv(ex / "typology_vs_store_type.csv")
    hist.to_csv(ex / "fact_typology_monthly.csv", index=False)
    forecast.to_csv(ex / "fact_forecast.csv", index=False)
    backtest.to_csv(ex / "fact_backtest.csv", index=False)
    bt_summary.to_csv(ex / "backtest_summary.csv")
    bt_by_typo.to_csv(ex / "backtest_by_typology.csv")
    core.to_csv(ex / "core_range.csv", index=False)
    missing.to_csv(ex / "fact_missing_items.csv", index=False)
    per_store.to_csv(ex / "fact_store_gaps.csv", index=False)
    fam_gaps.to_csv(ex / "fact_family_gaps.csv", index=False)
    joblib.dump({"pca": pca, "clustering": cl}, s.models / "segmentation.joblib")

    summary = {
        "n_stores_active": len(X),
        "stores_supplementary": [int(x) for x in excluded],
        "n_variables": int(X.shape[1]),
        "pca_n_components": int(pca.n_components),
        "pca_cum_variance_pct": round(
            float(pca.eigenvalues["cumulative_pct"].iloc[pca.n_components - 1]), 1
        ),
        "n_clusters": int(cl.n_clusters),
        "kmeans_moved": cl.moved,
        "ari_vs_store_type": round(ari, 3),
        "backtest": bt_summary.round(2).to_dict(orient="index"),
        "core_items_by_typology": core.groupby("typology").size().to_dict(),
        "mean_coverage_pct": round(float(per_store["coverage_pct"].mean()), 1),
        "potential_units_missing": round(float(per_store["potential_units_missing"].sum())),
        "potential_uplift_pct_on_core": round(
            float(
                per_store["potential_units_missing"].sum()
                / per_store["actual_units_core"].sum()
                * 100
            ),
            2,
        ),
    }
    (ex / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-extract", action="store_true", help="reutilise data/processed")
    args = parser.parse_args()
    run(get_settings(), skip_extract=args.skip_extract)


if __name__ == "__main__":
    main()
