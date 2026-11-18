import numpy as np
import pandas as pd
import pytest

from pca_segmentation.data.aggregation import typology_monthly
from pca_segmentation.data.cleaning import clean_monthly_sales
from pca_segmentation.features.build_features import family_shares, presence_index
from pca_segmentation.forecasting.backtest import mape, rmse, rolling_origin, summarize
from pca_segmentation.forecasting.baseline import seasonal_naive
from pca_segmentation.forecasting.model import holt_winters
from pca_segmentation.reporting.gap_analysis import store_gaps, theoretical_assortment


@pytest.fixture
def seasonal_series() -> pd.Series:
    idx = pd.date_range("2013-01-01", periods=48, freq="MS")
    season = 1 + 0.3 * (idx.month == 12) + 0.1 * np.sin(2 * np.pi * idx.month / 12)
    trend = 100 + 0.5 * np.arange(48)
    noise = np.random.default_rng(1).normal(0, 1, 48)
    return pd.Series(trend * season + noise, index=idx)


def test_seasonal_naive_repeats_last_year(seasonal_series):
    fc = seasonal_naive(seasonal_series, 3)
    assert list(fc.values) == list(seasonal_series.iloc[-12:-9].values)
    assert fc.index[0] == pd.Timestamp("2017-01-01")


def test_metrics():
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_backtest_has_no_leakage(seasonal_series):
    seen = []

    def spy(train, h):
        seen.append(train.index[-1])
        return seasonal_naive(train, h)

    bt = rolling_origin(seasonal_series, h=3, n_origins=5, models={"spy": spy})
    assert (bt["month"] > bt["origin"]).all()
    assert len(seen) == 5 and len(bt) == 15


def test_holt_winters_beats_baseline_on_trending_series(seasonal_series):
    bt = rolling_origin(
        seasonal_series, 3, 8, {"seasonal_naive": seasonal_naive, "holt_winters": holt_winters}
    )
    summary = summarize(bt)
    assert summary.loc["holt_winters", "mape"] < summary.loc["seasonal_naive", "mape"]


def test_cleaning_clips_returns_and_dedups():
    df = pd.DataFrame(
        {
            "store_nbr": [1, 1, 1],
            "family": ["A", "A", "B"],
            "month": ["2017-01-01", "2017-01-01", "2017-01-15"],
            "units": [5.0, -2.0, None],
            "n_items_sold": [1, 1, 0],
            "units_promo": [0, 0, 0],
        }
    )
    out = clean_monthly_sales(df)
    assert len(out) == 2
    assert (out["units"] >= 0).all()
    assert out["month"].dt.day.eq(1).all()


def test_features_shares_and_presence():
    monthly = pd.DataFrame(
        {
            "store_nbr": [1, 1, 2],
            "family": ["A", "B", "A"],
            "month": "2017-01-01",
            "units": [30.0, 70.0, 10.0],
        }
    )
    shares = family_shares(monthly, ["A", "B"])
    assert np.allclose(shares.sum(axis=1), 100)
    items = pd.DataFrame(
        {"store_nbr": [1, 1, 2], "item_nbr": [10, 11, 10], "family": ["A", "A", "A"]}
    )
    pres = presence_index(items, ["A"])
    assert pres.loc[1, "A"] == 100 and pres.loc[2, "A"] == 50


def test_typology_monthly_normalizes_by_active_stores():
    monthly = pd.DataFrame(
        {
            "store_nbr": [1, 1, 2],
            "month": pd.to_datetime(["2017-01-01", "2017-02-01", "2017-02-01"]),
            "units": [100.0, 100.0, 100.0],
        }
    )
    out = typology_monthly(monthly, pd.Series({1: 1, 2: 1}))
    # l'ouverture du magasin 2 en fevrier ne doit pas apparaitre comme une croissance
    assert out["units_per_store"].tolist() == [100.0, 100.0]


def test_gap_analysis():
    store_items = pd.DataFrame(
        {
            "store_nbr": [1, 2, 3, 1, 2, 3, 3],
            "item_nbr": [10, 10, 10, 11, 11, 99, 12],
            "family": ["A"] * 7,
            "units": [5.0, 7.0, 9.0, 4.0, 6.0, 1.0, 2.0],
        }
    )
    labels = pd.Series({1: 1, 2: 1, 3: 1}, name="typology")
    core = theoretical_assortment(store_items, labels, threshold=0.6)
    assert set(core["item_nbr"]) == {10, 11}
    missing, per_store = store_gaps(store_items, labels, core)
    assert missing[["store_nbr", "item_nbr"]].values.tolist() == [[3, 11]]
    assert missing["potential_units"].iloc[0] == pytest.approx(5.0)  # mediane de 4 et 6
    row = per_store.set_index("store_nbr").loc[3]
    assert row["coverage_pct"] == pytest.approx(50.0)
    assert row["off_range_items"] == 2
