"""Momentum signal: ranking direction, threshold, calendar."""
import numpy as np
import pandas as pd

from src.signals import calc_vol, generate_signal


def _exp_growth_prices(growth: list[float]) -> pd.DataFrame:
    """Each ticker grows at its own constant daily rate -> momentum order is known for every window."""
    idx = pd.bdate_range("2015-01-01", periods=800)
    t = np.arange(len(idx))[:, None]
    return pd.DataFrame(100 * np.exp(t * np.array(growth)), index=idx,
                        columns=[f"G{i}" for i in range(len(growth))])


def test_rank_1_is_strongest_momentum():
    growth = [0.0001 * i for i in range(12)]  # G11 strongest, G0 weakest
    sig = generate_signal(_exp_growth_prices(growth), 1, 3, 12, 1 / 3, 1 / 3, 1 / 3, thresh=10)

    assert len(sig) > 0
    expected = pd.Series(np.arange(12, 0, -1, dtype=float), index=sig.columns)  # G0 -> 12 ... G11 -> 1
    for _, row in sig.iterrows():
        pd.testing.assert_series_equal(row, expected, check_names=False)


def test_signal_is_on_business_month_ends(prices):
    sig = generate_signal(prices)
    assert all(d == d + pd.offsets.BMonthEnd(0) for d in sig.index)


def test_rows_below_thresh_are_dropped(prices_late):
    # 9 full-history tickers + 3 starting 2016-01; long_window=12 -> late tickers get a
    # signal only from 2017-01 month-end, so until then only 9 < thresh=10 tickers are valid.
    sig = generate_signal(prices_late, 1, 3, 12, 1 / 3, 1 / 3, 1 / 3, thresh=10)

    assert sig.index[0] == pd.Timestamp("2017-01-31")
    assert sig.notna().sum(axis=1).min() >= 10


def test_ranks_are_a_permutation_per_row(prices_late):
    sig = generate_signal(prices_late, thresh=9)
    for _, row in sig.iterrows():
        valid = row.dropna()
        assert sorted(valid) == list(range(1, len(valid) + 1))


def test_vol_of_constant_growth_is_zero():
    p = _exp_growth_prices([0.001, 0.002])
    vol = calc_vol(p).dropna()
    assert np.allclose(vol.values, 0.0, atol=1e-10)
