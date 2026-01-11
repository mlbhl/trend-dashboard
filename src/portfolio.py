"""Portfolio weight computation and backtesting functions."""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BMonthBegin

from .signals import calc_vol


def compute_weight_top_k(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    top_k: int = 5,
    inverse_vol: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute portfolio weights selecting top-k ranked assets.

    Args:
        price: Daily price DataFrame
        signal: Signal DataFrame with rankings (1 = best)
        top_k: Number of top-ranked assets to select
        inverse_vol: If True, weight by inverse volatility

    Returns:
        Tuple of (portfolio weights, benchmark weights)
    """
    vol = calc_vol(price).resample('BM').last()

    wgt = pd.DataFrame(0., index=signal.index, columns=signal.columns)
    bm_wgt = pd.DataFrame(0., index=signal.index, columns=signal.columns)

    for date in signal.index:
        row = signal.loc[date]
        valid = row[~row.isna()].index

        if len(valid) > 0:
            bm_wgt.loc[date, valid] = 1.0 / len(valid)

        selected = row[row <= top_k].index
        if len(selected) == 0:
            continue

        if inverse_vol:
            inv_vol = 1 / vol.loc[date, selected]
            inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
            w = inv_vol / inv_vol.sum()
        else:
            w = pd.Series(1.0 / len(selected), index=selected)

        wgt.loc[date, selected] = w

    return wgt, bm_wgt


def compute_weight_quantile(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    n_quantiles: int = 5,
    inverse_vol: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Compute portfolio weights for each quantile.

    Args:
        price: Daily price DataFrame
        signal: Signal DataFrame with rankings
        n_quantiles: Number of quantile groups
        inverse_vol: If True, weight by inverse volatility

    Returns:
        Tuple of (dict of quantile weights, benchmark weights)
    """
    vol = calc_vol(price).resample('BM').last()

    bm_wgt = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    q_wgts = {
        f"Q{q}": pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
        for q in range(1, n_quantiles + 1)
    }

    for date in signal.index:
        row = signal.loc[date]
        valid = row[~row.isna()].index

        if len(valid) == 0:
            continue

        bm_wgt.loc[date, valid] = 1.0 / len(valid)

        s = row.loc[valid].copy()
        if s.nunique() == 1:
            continue

        qlabel = pd.qcut(
            s.rank(method="first", ascending=False),
            q=n_quantiles,
            labels=[f"Q{i}" for i in range(1, n_quantiles + 1)]
        )

        for q in qlabel.unique():
            selected = qlabel[qlabel == q].index
            if len(selected) == 0:
                continue

            if inverse_vol:
                inv_vol = 1.0 / vol.loc[date, selected]
                inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
                w = inv_vol / inv_vol.sum()
            else:
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

    ret = price.pct_change()
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
                L_prev = prev_w.sum()
                dollar_pnl = dollar_pos * ret_sub.loc[date].values
                dollar_pos += dollar_pnl
                prev_w = (dollar_pos / dollar_pos.sum()) * L_prev
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
            L_prev = prev_w.sum()
            dollar_pnl = dollar_pos * ret_sub.loc[date].values
            dollar_pos += dollar_pnl
            nav.loc[date] = prev_equity + dollar_pnl.sum()
            prev_w = (dollar_pos / dollar_pos.sum()) * L_prev

    turnover = pd.Series(turnover)
    return nav, turnover


def run_quantile_backtest(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    n_quantiles: int = 5,
    capital: float = 1000,
    tcost: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Run backtest for all quantiles and benchmark.

    Returns:
        Tuple of (NAV DataFrame, turnover DataFrame, benchmark NAV)
    """
    q_wgts, bm_wgt = compute_weight_quantile(price, signal, n_quantiles)

    # Shift weights to next month beginning
    bm_wgt.index = bm_wgt.index + BMonthBegin(1)
    bm_nav, _ = backtest(price, bm_wgt, capital=capital, tcost=tcost)
    bm_nav.name = 'BM'

    q_nav = {}
    q_to = {}
    for q in range(1, n_quantiles + 1):
        wgt = q_wgts[f"Q{q}"]
        wgt.index = wgt.index + BMonthBegin(1)
        nav, turnover = backtest(price, wgt, capital=capital, tcost=tcost)
        q_nav[f"Q{q}"] = nav
        q_to[f"Q{q}"] = turnover

    q_nav_df = pd.DataFrame(q_nav)
    q_to_df = pd.DataFrame(q_to)

    return q_nav_df, q_to_df, bm_nav


def run_top_k_backtest(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    top_k: int = 5,
    inverse_vol: bool = False,
    capital: float = 1000,
    tcost: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Run backtest for top-k strategy and benchmark.

    Returns:
        Tuple of (strategy NAV, benchmark NAV, turnover)
    """
    top_wgt, bm_wgt = compute_weight_top_k(price, signal, top_k, inverse_vol)

    # Shift weights to next month beginning
    bm_wgt.index = bm_wgt.index + BMonthBegin(1)
    bm_nav, _ = backtest(price, bm_wgt, capital=capital, tcost=tcost)

    top_wgt.index = top_wgt.index + BMonthBegin(1)
    top_nav, turnover = backtest(price, top_wgt, capital=capital, tcost=tcost)

    return top_nav, bm_nav, turnover
