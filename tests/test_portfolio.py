"""Weights (top-k / quantile) and the backtest engine."""
import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import BDay, BMonthBegin

from src import config
from src.charts import create_holding_heatmap
from src.portfolio import (
    backtest,
    blend_core_satellite,
    compute_weight_quantile,
    compute_weight_top_k,
    holding_months,
    rebalance_dates,
    run_quantile_backtest,
    run_top_k_backtest,
    to_rebalance_dates,
)
from src.signals import calc_vol, generate_signal

from conftest import make_prices


# ---------------------------------------------------------------- top-k weights

@pytest.mark.parametrize("k", [3, 5])
def test_top_k_equal_selects_best_ranks(prices_late, k):
    sig = generate_signal(prices_late, thresh=9)
    wgt, bm = compute_weight_top_k(prices_late, sig, top_k=k)

    for date, row in sig.iterrows():
        held = wgt.loc[date][wgt.loc[date] > 0]
        assert set(held.index) == set(row[row <= k].index)
        assert np.allclose(held.values, 1 / k)

        valid = row.dropna().index
        assert np.allclose(bm.loc[date, valid].values, 1 / len(valid))
        assert bm.loc[date].sum() == pytest.approx(1.0)


def test_top_k_none_holds_every_valid_ticker(prices_late):
    sig = generate_signal(prices_late, thresh=9)
    wgt, bm = compute_weight_top_k(prices_late, sig, top_k=None)
    pd.testing.assert_frame_equal(wgt, bm)


def test_rank_weighting(prices):
    sig = generate_signal(prices)
    wgt, _ = compute_weight_top_k(prices, sig, top_k=5, weight_method="rank")
    expected = {1: 5 / 15, 2: 4 / 15, 3: 3 / 15, 4: 2 / 15, 5: 1 / 15}
    for date, row in sig.iterrows():
        for ticker, rank in row[row <= 5].items():
            assert wgt.loc[date, ticker] == pytest.approx(expected[int(rank)])


def test_inverse_vol_weighting(prices):
    sig = generate_signal(prices)
    wgt, _ = compute_weight_top_k(prices, sig, top_k=5, weight_method="inverse_vol")
    vol = calc_vol(prices).resample("BME").last()
    for date in sig.index:
        held = wgt.loc[date][wgt.loc[date] > 0]
        assert held.sum() == pytest.approx(1.0)
        risk = held * vol.loc[date, held.index]  # w_i * vol_i must be equal across holdings
        assert np.allclose(risk.values, risk.iloc[0])


def test_blend_core_satellite(prices):
    sig = generate_signal(prices)
    sat, _ = compute_weight_top_k(prices, sig, top_k=5)
    price_x = prices.assign(SPY=100.0)

    w = blend_core_satellite(sat, price_x, ["SPY", "T00", "MISSING"], [2, 1, 1], 0.6)

    assert list(w.columns) == list(price_x.columns)
    assert w["SPY"].eq(0.6 * 2 / 4).all()                                   # core share, normalised
    assert (w["T00"] - 0.6 * 1 / 4 - 0.4 * sat["T00"]).abs().max() < 1e-15  # core + satellite overlap
    assert np.allclose(w.sum(axis=1), 1 - 0.6 * 1 / 4)                      # missing core ticker = cash


# ---------------------------------------------------------------- quantile weights

def test_quantiles_partition_universe_best_in_top_bucket():
    p = make_prices(n_tickers=10)
    sig = generate_signal(p, thresh=10)
    q_wgts, _ = compute_weight_quantile(p, sig, n_quantiles=5)

    for date, row in sig.iterrows():
        members = {q: set(w.loc[date][w.loc[date] > 0].index) for q, w in q_wgts.items()}
        for q, w in q_wgts.items():
            assert w.loc[date].sum() == pytest.approx(1.0)
            assert len(members[q]) == 2
        assert set().union(*members.values()) == set(row.dropna().index)
        assert members["Q5"] == set(row[row <= 2].index)   # best two ranks
        assert members["Q1"] == set(row[row >= 9].index)   # worst two ranks


# ---------------------------------------------------------------- backtest engine

def _monthly_weights(prices: pd.DataFrame, alloc: dict[str, float]) -> pd.DataFrame:
    dates = prices.resample("BME").last().index + BMonthBegin(1)
    dates = dates[dates <= prices.index[-1]]
    w = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    for t, v in alloc.items():
        w[t] = v
    return w


def test_full_allocation_is_buy_and_hold_and_costs_nothing(prices):
    w = _monthly_weights(prices, {"T03": 1.0})
    nav, _ = backtest(prices, w, capital=1000, tcost=0.01)  # rebalancing 100% -> 100% has no turnover
    start = w.index[0]
    expected = 1000 * prices["T03"][start:] / prices["T03"][start]
    pd.testing.assert_series_equal(nav, expected, check_names=False, check_freq=False, rtol=1e-12)


def test_matches_reference_rebalanced_portfolio(prices):
    """Independent implementation: hold fixed shares between rebalances, reset to 60/40 at each rebalance close."""
    w = _monthly_weights(prices, {"T01": 0.6, "T07": 0.4})
    nav, _ = backtest(prices, w, capital=1000, tcost=0.0)

    p = prices[["T01", "T07"]][w.index[0]:]
    target = np.array([0.6, 0.4])
    ref = pd.Series(index=p.index, dtype=float)
    value, shares = 1000.0, None
    for date in p.index:
        if shares is not None:
            value = float(shares @ p.loc[date].values)
        if date in w.index:
            shares = value * target / p.loc[date].values
        ref[date] = value
    pd.testing.assert_series_equal(nav, ref, check_names=False, check_freq=False, rtol=1e-10)


def test_price_gap_return_is_not_lost(prices):
    """A missing quote mid-series is padded: the move across the gap is booked when quotes resume."""
    gappy = prices.copy()
    gappy.iloc[300:305, 1] = np.nan
    w = _monthly_weights(prices, {"T01": 1.0})

    nav, _ = backtest(prices, w, tcost=0.0)
    nav_gap, _ = backtest(gappy, w, tcost=0.0)
    assert nav_gap.iloc[-1] == pytest.approx(nav.iloc[-1], rel=1e-12)


def test_unlisted_ticker_in_window_is_harmless(prices):
    """Optimizer training windows can contain a ticker with no prices yet (all-NaN column)."""
    w = _monthly_weights(prices, {"T01": 0.5, "T02": 0.5})
    with_unlisted = prices.assign(NEW=np.nan)
    w_unlisted = w.assign(NEW=0.0)

    nav, _ = backtest(prices, w, tcost=0.002)
    nav_u, _ = backtest(with_unlisted, w_unlisted, tcost=0.002)  # must not warn (FutureWarning = error)
    pd.testing.assert_series_equal(nav_u, nav)


def test_switch_cost_is_turnover_times_two_tcost():
    idx = pd.bdate_range("2020-01-01", "2020-04-30")
    flat = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    d1, d2 = pd.Timestamp("2020-02-03"), pd.Timestamp("2020-03-02")
    w = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=[d1, d2])
    tcost = 0.002

    nav, turnover = backtest(flat, w, capital=1000, tcost=tcost)

    assert turnover[d2] == pytest.approx(1.0)
    assert nav[d2] == pytest.approx(1000 * (1 - 1.0 * tcost * 2))
    assert nav.iloc[-1] == pytest.approx(nav[d2])  # flat prices: nothing else changes NAV


# ---------------------------------------------------------------- implementation lag

def test_default_lag_rebalances_on_first_business_day_of_next_month(prices):
    assert config.IMPLEMENTATION_LAG == 1
    sig = generate_signal(prices)
    nav, bm_nav, _ = run_top_k_backtest(prices, sig, top_k=5)
    first = sig.index[0] + BMonthBegin(1)
    assert nav.index[0] == first and bm_nav.index[0] == first


@pytest.mark.parametrize("lag", [0, 1, 2, 5])
def test_every_rebalance_happens_lag_business_days_after_signal(prices, monkeypatch, lag):
    monkeypatch.setattr(config, "IMPLEMENTATION_LAG", lag)
    sig = generate_signal(prices)
    expected = sig.index + BDay(lag)
    expected = list(expected[expected <= prices.index[-1]])

    nav, bm_nav, turnover = run_top_k_backtest(prices, sig, top_k=5, tcost=0.002)
    _, q_turnover, q_bm_nav = run_quantile_backtest(prices, sig, n_quantiles=5)

    assert list(turnover.index) == expected
    assert list(q_turnover.index) == expected
    assert nav.index[0] == bm_nav.index[0] == q_bm_nav.index[0] == expected[0]


def test_no_hardcoded_rebalance_offsets_outside_helper():
    """The lag must come only from portfolio.rebalance_dates(); a stray `+ BMonthBegin(1)` or
    `+ BDay(n)` elsewhere would silently ignore config.IMPLEMENTATION_LAG."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"\+\s*(pd\.offsets\.|pd\.tseries\.offsets\.)?(BMonthBegin|BDay|BusinessDay)\(")
    offenders = []
    for f in [*root.glob("src/*.py"), *root.glob("dash_components/*.py"), *root.glob("scripts/*.py")]:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if pattern.search(line) and not (f.name == "portfolio.py" and "BDay(lag)" in line):
                offenders.append(f"{f.relative_to(root)}:{n}: {line.strip()}")
    assert not offenders, "use portfolio.to_rebalance_dates()/rebalance_dates():\n" + "\n".join(offenders)


@pytest.mark.parametrize("lag", [0, 1, 2, 5])
def test_holding_month_is_the_month_after_the_signal(lag):
    """Lag 0 buys at the signal-day close, lag 1+ early next month: either way the portfolio is
    held through the month after the signal."""
    signals = pd.date_range("2024-01-01", periods=24, freq="BME")
    months = holding_months(rebalance_dates(signals, lag), lag)
    assert list(months) == [s + pd.offsets.MonthBegin(1) for s in signals]


def test_holdings_heatmap_labels_do_not_depend_on_lag(prices, monkeypatch):
    sig = generate_signal(prices)
    labels = {}
    for lag in (0, 1, 2):
        monkeypatch.setattr(config, "IMPLEMENTATION_LAG", lag)
        w = to_rebalance_dates(compute_weight_top_k(prices, sig, top_k=5)[0])
        labels[lag] = list(create_holding_heatmap(w.iloc[-6:]).layout.xaxis.ticktext)
    assert labels[0] == labels[1] == labels[2]
    assert labels[1][-1] == (sig.index[-1] + pd.offsets.MonthBegin(1)).strftime("%y.%m")


def test_lag_changes_results(prices, monkeypatch):
    sig = generate_signal(prices)
    finals = {}
    for lag in (0, 1, 3):
        monkeypatch.setattr(config, "IMPLEMENTATION_LAG", lag)
        finals[lag] = run_top_k_backtest(prices, sig, top_k=5)[0].iloc[-1]
    assert len(set(finals.values())) == 3
