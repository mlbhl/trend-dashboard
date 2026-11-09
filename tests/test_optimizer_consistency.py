"""The optimizer must score parameters with exactly the dashboard's rules.

Signals come from a momentum cache (combine_signals) and NAVs are cut into train/test
windows; these tests pin both to the dashboard path (generate_signal, portfolio.backtest)."""
import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    _benchmark_nav,
    _blend_core_weights,
    _calc_sharpe_from_nav,
    _strategy_nav,
    _strategy_sharpe,
    combine_signals,
    precompute_momentums,
)
from src import config
from src.portfolio import (
    backtest,
    compute_weight_core,
    compute_weight_top_k,
    run_top_k_backtest,
    to_rebalance_dates,
)
from src.signals import calc_vol, generate_signal

PARAMS = dict(short_window=1, mid_window=3, long_window=11, short_wgt=0.1, mid_wgt=0.1, long_wgt=0.8)


def test_combine_signals_equals_generate_signal(prices_late):
    fast = combine_signals(precompute_momentums(prices_late), **PARAMS, thresh=9)
    slow = generate_signal(prices_late, **PARAMS, thresh=9)
    pd.testing.assert_frame_equal(fast, slow)


@pytest.mark.parametrize("lag", [0, 1, 2])
@pytest.mark.parametrize("weight_method", ["equal", "inverse_vol", "rank"])
def test_strategy_nav_and_sharpe_equal_dashboard_backtest(prices_late, monkeypatch, weight_method, lag):
    monkeypatch.setattr(config, "IMPLEMENTATION_LAG", lag)
    sig = generate_signal(prices_late, **PARAMS, thresh=9)
    vol = calc_vol(prices_late).resample("BME").last()

    nav, _, _ = run_top_k_backtest(prices_late, sig, top_k=5, weight_method=weight_method,
                                   capital=1000, tcost=0.002)
    opt_nav = _strategy_nav(prices_late, sig, vol, top_k=5, weight_method=weight_method, tcost=0.002)
    opt_sharpe = _strategy_sharpe(prices_late, sig, vol, top_k=5, weight_method=weight_method, tcost=0.002)

    pd.testing.assert_series_equal(opt_nav, nav, check_names=False, check_freq=False, rtol=1e-12)
    assert opt_sharpe == pytest.approx(_calc_sharpe_from_nav(nav), rel=1e-12)


@pytest.mark.parametrize("weight_method", ["equal", "inverse_vol"])
def test_training_window_score_equals_dashboard_run_on_that_window(prices_late, weight_method):
    """Optimizer scores a window on a price slice; it must still use full-history volatility,
    i.e. match the dashboard run on full data restarted at the window start."""
    start, end = "2017-06-01", "2018-12-31"
    sig = generate_signal(prices_late, **PARAMS, thresh=9)
    vol = calc_vol(prices_late).resample("BME").last()

    wgt, _ = compute_weight_top_k(prices_late, sig, top_k=5, weight_method=weight_method)
    wgt = to_rebalance_dates(wgt)
    dash_nav, _ = backtest(prices_late[:end], wgt, capital=1000, tcost=0.002, start_date=start)

    score = _strategy_sharpe(prices_late[start:end], sig, vol, top_k=5, weight_method=weight_method,
                             tcost=0.002, start_date=start)
    assert score == pytest.approx(_calc_sharpe_from_nav(dash_nav), rel=1e-12)


def test_window_nav_restarts_at_first_rebalance_and_stops_at_end(prices):
    sig = generate_signal(prices, **PARAMS)
    vol = calc_vol(prices).resample("BME").last()
    nav = _strategy_nav(prices, sig, vol, start_date="2017-03-15", end_date="2017-09-30")

    assert nav.index[0] == pd.Timestamp("2017-04-03")  # first rebalance on/after start
    assert nav.index[-1] == pd.Timestamp("2017-09-29")  # last business day <= end
    assert nav.iloc[0] == 1000.0
    assert _strategy_nav(prices, sig, vol, start_date="2030-01-01") is None


@pytest.mark.parametrize("lag", [0, 1, 2])
def test_benchmark_nav_equals_dashboard_benchmark(prices, monkeypatch, lag):
    monkeypatch.setattr(config, "IMPLEMENTATION_LAG", lag)
    sig = generate_signal(prices, **PARAMS)
    _, bm_wgt = compute_weight_top_k(prices, sig, top_k=5)
    bm_wgt = to_rebalance_dates(bm_wgt)
    start = str(bm_wgt.index[3].date())

    bm_nav, _ = backtest(prices, bm_wgt, capital=1000, tcost=0, start_date=start)
    opt_bm = _benchmark_nav(prices, tcost=0, start_date=start)
    pd.testing.assert_series_equal(opt_bm, bm_nav, check_names=False, check_freq=False, rtol=1e-12)


def test_core_blend_matches_dashboard_formula(prices):
    sig = generate_signal(prices, **PARAMS)
    sat, _ = compute_weight_top_k(prices, sig, top_k=5)
    cfg = {"tickers": ["T00", "T01", "NEW"], "weights": [2, 1, 1], "ratio": 0.7}

    blended, price_x = _blend_core_weights(sat, prices, cfg)

    assert list(price_x.columns) == list(prices.columns) + ["NEW"]
    core = compute_weight_core(price_x, sig, cfg["tickers"], cfg["weights"])
    expected = 0.7 * core + 0.3 * sat.reindex(columns=price_x.columns, fill_value=0.0)
    pd.testing.assert_frame_equal(blended, expected)
    assert np.allclose(blended.sum(axis=1), 1.0)
