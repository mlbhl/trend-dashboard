"""Portfolio weight computation and backtesting functions."""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from . import config
from .signals import calc_vol


def rebalance_dates(signal_dates, lag: int | None = None) -> pd.DatetimeIndex:
    """Rebalance date for each month-end signal date: `lag` business days later.

    `lag` defaults to config.IMPLEMENTATION_LAG. Every weight series (backtests, holdings
    displays, optimizer) goes through here so they all share one lag.
    """
    if lag is None:
        lag = config.IMPLEMENTATION_LAG
    return pd.DatetimeIndex(signal_dates) + BDay(lag)


def to_rebalance_dates(weights: pd.DataFrame, lag: int | None = None) -> pd.DataFrame:
    """Copy of signal-date (month-end) weights, re-indexed to their rebalance dates."""
    shifted = weights.copy()
    shifted.index = rebalance_dates(weights.index, lag)
    return shifted


def compute_weight_core(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    core_tickers: list[str],
    core_weights: list[float],
) -> pd.DataFrame:
    """
    Compute fixed monthly weights for core holdings.

    Args:
        price: Daily price DataFrame (used only for column alignment)
        signal: Signal DataFrame (used for month-end index alignment)
        core_tickers: List of ticker symbols for core holdings
        core_weights: List of weights (will be normalized to sum to 1)

    Returns:
        DataFrame of core weights indexed by signal dates, columns matching price
    """
    total = sum(core_weights)
    norm_weights = [w / total for w in core_weights] if total > 0 else core_weights

    wgt = pd.DataFrame(0.0, index=signal.index, columns=price.columns)
    for ticker, w in zip(core_tickers, norm_weights):
        if ticker in wgt.columns:
            wgt[ticker] = w

    return wgt


def equal_weight_benchmark(signal: pd.DataFrame) -> pd.DataFrame:
    """Benchmark weights: equal weight across every ticker with a signal on each date.

    Same candidate set the strategy picks from, so strategy-vs-BM isolates selection skill.
    Rows with no signal stay 0.
    """
    has_signal = signal.notna()
    return has_signal.div(has_signal.sum(axis=1), axis=0).fillna(0.0)


def blend_core_satellite(
    sat_wgt: pd.DataFrame,
    price: pd.DataFrame,
    core_tickers: list[str],
    core_weights: list[float],
    core_ratio: float,
) -> pd.DataFrame:
    """Core-satellite weights on price's columns: core_ratio * fixed core + (1 - core_ratio) * satellite.

    sat_wgt is indexed by signal date. Core tickers missing from price are dropped (held as cash).
    """
    core = compute_weight_core(price, sat_wgt, core_tickers, core_weights)
    sat = sat_wgt.reindex(columns=price.columns, fill_value=0.0)
    return core_ratio * core + (1 - core_ratio) * sat


def compute_weight_top_k(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    top_k: int | None = 5,
    weight_method: str = "equal",
    vol: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute portfolio weights selecting top-k ranked assets.

    Args:
        price: Daily price DataFrame
        signal: Signal DataFrame with rankings (1 = best)
        top_k: Number of top-ranked assets to select (None for all)
        weight_method: Weighting method - "equal", "inverse_vol", or "rank"
        vol: Precomputed month-end volatility; computed from `price` when None.
            The optimizer passes full-history vol while `price` is a training-window slice.

    Returns:
        Tuple of (portfolio weights, benchmark weights)
    """
    if vol is None:
        vol = calc_vol(price).resample('BME').last()

    wgt = pd.DataFrame(0., index=signal.index, columns=signal.columns)
    bm_wgt = equal_weight_benchmark(signal)

    for date in signal.index:
        row = signal.loc[date]
        valid = row[~row.isna()].index

        # Select top-k or all valid tickers
        if top_k is None:
            selected = valid
        else:
            selected = row[row <= top_k].index

        if len(selected) == 0:
            continue

        if weight_method == "inverse_vol":
            inv_vol = 1 / vol.loc[date, selected]
            inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
            if len(inv_vol) == 0:
                continue
            w = inv_vol / inv_vol.sum()
        elif weight_method == "rank":
            # Weight by rank (best ticker gets highest rank)
            # Convert: rank 1 (best) -> len(selected), rank len(selected) (worst) -> 1
            ranks = row.loc[selected]
            converted_rank = len(selected) + 1 - ranks
            w = converted_rank / converted_rank.sum()
        else:  # equal
            w = pd.Series(1.0 / len(selected), index=selected)

        wgt.loc[date, selected] = w

    return wgt, bm_wgt


def compute_weight_quantile(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    n_quantiles: int = 5,
    weight_method: str = "equal",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Compute portfolio weights for each quantile.

    Args:
        price: Daily price DataFrame
        signal: Signal DataFrame with rankings
        n_quantiles: Number of quantile groups
        weight_method: Weighting method - "equal", "inverse_vol", or "rank"

    Returns:
        Tuple of (dict of quantile weights, benchmark weights)
    """
    vol = calc_vol(price).resample('BME').last()

    bm_wgt = equal_weight_benchmark(signal)
    q_wgts = {
        f"Q{q}": pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
        for q in range(1, n_quantiles + 1)
    }

    for date in signal.index:
        row = signal.loc[date]
        valid = row[~row.isna()].index

        if len(valid) == 0:
            continue

        s = row.loc[valid].copy()
        if s.nunique() == 1:
            continue

        pct = s.rank(pct=True)
        bins = np.linspace(0, 1, n_quantiles + 1)
        qlabel = pd.cut(
            pct,
            bins=bins,
            labels=[f"Q{i}" for i in range(n_quantiles, 0, -1)],
            include_lowest=True,
        )

        for q in qlabel.unique():
            selected = qlabel[qlabel == q].index
            if len(selected) == 0:
                continue

            if weight_method == "inverse_vol":
                inv_vol = 1.0 / vol.loc[date, selected]
                inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
                w = inv_vol / inv_vol.sum()
            elif weight_method == "rank":
                # Weight by rank within quantile (best ticker gets highest rank)
                ranks = row.loc[selected]
                converted_rank = len(selected) + 1 - ranks.rank(method="first")
                w = converted_rank / converted_rank.sum()
            else:  # equal
                w = pd.Series(1.0 / len(selected), index=selected)

            q_wgts[str(q)].loc[date, selected] = w

    return q_wgts, bm_wgt


def backtest(
    price: pd.DataFrame,
    weight: pd.DataFrame,
    capital: float = 1000,
    tcost: float = 0.002,
    start_date: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Run backtest with given weights.

    Args:
        price: Daily price DataFrame
        weight: Portfolio weights DataFrame (monthly)
        capital: Initial capital
        tcost: Transaction cost (one-way)
        start_date: Optional start date for backtest

    Returns:
        Tuple of (NAV series, turnover series)
    """
    if start_date is None:
        start_date = weight.index[0]
        weight_dates = weight.index
    else:
        start_date = weight[start_date:].index[0]
        weight_dates = weight[start_date:].index

    # Explicit ffill == pandas' legacy pct_change() padding, without its deprecation warning
    # (raised when a column is all-NaN, e.g. a ticker not yet listed in an optimizer training window)
    ret = price.ffill().pct_change(fill_method=None)
    ret_sub = ret[start_date:].fillna(0.)
    all_dates = ret_sub.index

    nav = pd.Series(index=all_dates, dtype=float)
    turnover = {}
    nav.loc[start_date] = capital
    dollar_pos = None
    prev_w = None

    for i, date in enumerate(all_dates):
        if date in weight_dates:
            if date != start_date:
                prev_date = all_dates[i - 1]
                prev_equity = nav.loc[prev_date]
                dollar_pnl = dollar_pos * ret_sub.loc[date].values
                dollar_pos += dollar_pnl
                prev_w = dollar_pos / dollar_pos.sum()
                new_w = weight.loc[date].values
                turn = np.sum(np.abs(new_w - prev_w)) / 2
                equity_after_cost = (prev_equity + dollar_pnl.sum()) * (1 - turn * tcost * 2)
            else:
                new_w = weight.loc[date].values
                turn = np.sum(np.abs(new_w)) / 2
                equity_after_cost = capital

            nav.loc[date] = equity_after_cost
            dollar_pos = new_w * equity_after_cost
            turnover[date] = turn
            prev_w = new_w.copy()
        else:
            prev_date = all_dates[i - 1]
            prev_equity = nav.loc[prev_date]
            dollar_pnl = dollar_pos * ret_sub.loc[date].values
            dollar_pos += dollar_pnl
            nav.loc[date] = prev_equity + dollar_pnl.sum()
            prev_w = dollar_pos / dollar_pos.sum()

    turnover = pd.Series(turnover)
    return nav, turnover


def run_quantile_backtest(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    n_quantiles: int = 5,
    capital: float = 1000,
    tcost: float = 0.0,
    weight_method: str = "equal",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Run backtest for all quantiles and benchmark.

    Returns:
        Tuple of (NAV DataFrame, turnover DataFrame, benchmark NAV)
    """
    q_wgts, bm_wgt = compute_weight_quantile(price, signal, n_quantiles, weight_method)

    # Signal-date weights -> rebalance dates (config.IMPLEMENTATION_LAG)
    bm_wgt = to_rebalance_dates(bm_wgt)
    bm_nav, _ = backtest(price, bm_wgt, capital=capital, tcost=0)
    bm_nav.name = 'BM'

    q_nav = {}
    q_to = {}
    for q in range(1, n_quantiles + 1):
        wgt = to_rebalance_dates(q_wgts[f"Q{q}"])
        nav, turnover = backtest(price, wgt, capital=capital, tcost=tcost)
        q_nav[f"Q{q}"] = nav
        q_to[f"Q{q}"] = turnover

    q_nav_df = pd.DataFrame(q_nav)
    q_to_df = pd.DataFrame(q_to)

    return q_nav_df, q_to_df, bm_nav


def run_top_k_backtest(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    top_k: int | None = 5,
    weight_method: str = "equal",
    capital: float = 1000,
    tcost: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Run backtest for top-k strategy and benchmark.

    Args:
        price: Daily price DataFrame
        signal: Signal DataFrame with rankings
        top_k: Number of top-ranked assets (None for all)
        weight_method: Weighting method - "equal", "inverse_vol", or "rank"

    Returns:
        Tuple of (strategy NAV, benchmark NAV, turnover)
    """
    top_wgt, bm_wgt = compute_weight_top_k(price, signal, top_k, weight_method)

    # Signal-date weights -> rebalance dates (config.IMPLEMENTATION_LAG)
    bm_wgt = to_rebalance_dates(bm_wgt)
    bm_nav, _ = backtest(price, bm_wgt, capital=capital, tcost=0)

    top_wgt = to_rebalance_dates(top_wgt)
    top_nav, turnover = backtest(price, top_wgt, capital=capital, tcost=tcost)

    return top_nav, bm_nav, turnover
