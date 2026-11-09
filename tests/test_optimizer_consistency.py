"""The optimizer must score parameters with exactly the dashboard's rules.

Signals come from a momentum cache (combine_signals) and NAVs are cut into train/test
windows; these tests pin both to the dashboard path (generate_signal, portfolio.backtest)."""
import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    _benchmark_nav,
    _calc_sharpe_from_nav,
    _strategy_nav,
    _strategy_sharpe,
    combine_signals,
    optimize_sharpe,
    precompute_momentums,
    walk_forward_optimize,
)
from src import config
from src.portfolio import (
    backtest,
    blend_core_satellite,
    compute_weight_top_k,
    equal_weight_benchmark,
    run_top_k_backtest,
    to_rebalance_dates,
)

from conftest import make_prices
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
def test_benchmark_nav_equals_dashboard_benchmark(prices_late, monkeypatch, lag):
    """BM = equal weight across tickers with a signal. 2016-03: the 3 late tickers already have
    prices but no signal yet — they must not be in the benchmark."""
    monkeypatch.setattr(config, "IMPLEMENTATION_LAG", lag)
    sig = generate_signal(prices_late, **PARAMS, thresh=9)
    start = "2016-03-01"
    late = list(prices_late.columns[-3:])
    assert prices_late.loc["2016-03-01":"2016-03-31", late].notna().all().all()
    assert sig.loc["2016-03-31", late].isna().all()

    bm_wgt = to_rebalance_dates(equal_weight_benchmark(sig))
    dash_bm, _ = backtest(prices_late, bm_wgt, capital=1000, tcost=0, start_date=start)
    opt_bm = _benchmark_nav(prices_late, sig, tcost=0, start_date=start)
    pd.testing.assert_series_equal(opt_bm, dash_bm, check_names=False, check_freq=False, rtol=1e-12)


# ---------------------------------------------------------------- core-satellite universe

CORE = {"tickers": ["CORE_A", "CORE_B"], "weights": [60, 40], "ratio": 0.7}


def _with_core_tickers(universe_prices: pd.DataFrame) -> pd.DataFrame:
    """Add two non-selectable core tickers; CORE_A trends hard so momentum would pick it if allowed."""
    core = make_prices(n_tickers=2, seed=7).rename(columns={"T00": "CORE_A", "T01": "CORE_B"})
    core["CORE_A"] = core["CORE_A"] * np.exp(np.arange(len(core)) * 0.002)
    return universe_prices.join(core)


def _dashboard_core_satellite(price_all, universe, params, top_k, weight_method, tcost, start):
    """Mirror of the dashboard's Run Analysis in core-satellite mode."""
    sig = generate_signal(price_all[universe], **params, thresh=10)
    sat, _ = compute_weight_top_k(price_all[universe], sig, top_k=top_k, weight_method=weight_method)
    comb = blend_core_satellite(sat, price_all, CORE["tickers"], CORE["weights"], CORE["ratio"])
    nav, _ = backtest(price_all, to_rebalance_dates(comb), capital=1000, tcost=tcost, start_date=start)
    return nav, sat


@pytest.mark.parametrize("weight_method", ["equal", "inverse_vol"])
def test_core_satellite_optimizer_scores_the_dashboard_strategy(prices, weight_method):
    universe = list(prices.columns)
    price_all = _with_core_tickers(prices)
    start = "2016-06-01"

    res = optimize_sharpe(price_all, top_k=3, weight_method=weight_method, tcost=0.002, start_date=start,
                          n_samples=6, seed=0, core_config=CORE, universe=universe)
    nav, sat = _dashboard_core_satellite(price_all, universe, res["best_params"], 3, weight_method, 0.002, start)

    assert res["best_sharpe"] == pytest.approx(_calc_sharpe_from_nav(nav), rel=1e-12)
    assert set(sat.columns) == set(universe)  # core tickers never compete for satellite slots


@pytest.mark.parametrize("with_core", [False, True])
def test_walk_forward_oos_benchmark_equals_dashboard_benchmark(with_core):
    """Two tickers list in 2017-06, inside the 2018 test window's look-back: they have prices
    but no signal during part of the window, so the benchmark definition matters."""
    universe_prices = make_prices(late_tickers=2, late_start="2017-06-01")
    universe = list(universe_prices.columns)
    price_all = _with_core_tickers(universe_prices) if with_core else universe_prices
    core_cfg = CORE if with_core else None

    wf = walk_forward_optimize(price_all, train_months=24, test_months=12, step_months=12,
                               n_samples=3, seed=0, core_config=core_cfg, universe=universe)
    combined_ret = wf["combined_oos_bm_nav"].pct_change()
    assert len(wf["folds"]) >= 2

    for fold in wf["folds"]:
        sig = generate_signal(price_all[universe], **fold["params"], thresh=10)
        bm = equal_weight_benchmark(sig)
        bm = (blend_core_satellite(bm, price_all, CORE["tickers"], CORE["weights"], CORE["ratio"])
              if with_core else bm)
        dash_bm, _ = backtest(price_all[:fold["test_end"]], to_rebalance_dates(bm), capital=1000, tcost=0,
                              start_date=fold["test_start"])
        dash_ret = dash_bm.pct_change().dropna()
        np.testing.assert_allclose(combined_ret.reindex(dash_ret.index).values, dash_ret.values, rtol=1e-10)


