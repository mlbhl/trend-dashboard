"""Optimizer for momentum strategy parameters."""

import random
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BMonthBegin

from .signals import calc_vol


def precompute_momentums(price: pd.DataFrame, max_window: int = 12) -> dict[int, pd.DataFrame]:
    """
    Precompute momentum ranks for all windows (1 to max_window).

    Returns:
        Dict mapping window -> momentum rank DataFrame
    """
    price_m = price.resample('BM').last()
    momentums = {}

    for window in range(1, max_window + 1):
        mom = (price_m / price_m.shift(window)).rank(axis=1)
        momentums[window] = mom

    return momentums


def generate_lookback_combinations() -> list[tuple[int, int, int]]:
    """Generate all valid (short, mid, long) combinations where short < mid < long."""
    combinations = []
    for short in range(1, 11):  # short: 1-10
        for mid in range(short + 1, 12):  # mid: short+1 to 11
            for long in range(mid + 1, 13):  # long: mid+1 to 12
                combinations.append((short, mid, long))
    return combinations


def generate_weight_combinations(step: int = 10) -> list[tuple[float, float, float]]:
    """Generate all weight combinations summing to 100% with given step."""
    combinations = []
    values = list(range(0, 101, step))  # 0, 10, 20, ..., 100

    for w1 in values:
        for w2 in values:
            w3 = 100 - w1 - w2
            if 0 <= w3 <= 100:
                combinations.append((w1 / 100, w2 / 100, w3 / 100))

    return combinations


def combine_signals(
    momentums: dict[int, pd.DataFrame],
    short_window: int,
    mid_window: int,
    long_window: int,
    short_wgt: float,
    mid_wgt: float,
    long_wgt: float,
    thresh: int = 10,
) -> pd.DataFrame:
    """Combine precomputed momentums with weights to generate signal."""
    mom_s = momentums[short_window]
    mom_m = momentums[mid_window]
    mom_l = momentums[long_window]

    sig = (mom_s * short_wgt + mom_m * mid_wgt + mom_l * long_wgt).rank(
        axis=1, method="first", ascending=False
    )
    sig = sig.dropna(thresh=thresh)

    return sig


def fast_backtest_sharpe(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    vol: pd.DataFrame,
    top_k: int | None = 5,
    weight_method: str = "equal",
    tcost: float = 0.0,
    start_date: str | None = None,
) -> float:
    """
    Fast backtest returning only Sharpe ratio.
    Optimized version that skips unnecessary calculations.
    """
    # Compute weights
    wgt = pd.DataFrame(0., index=signal.index, columns=signal.columns)

    for date in signal.index:
        row = signal.loc[date]
        valid = row[~row.isna()].index

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
            ranks = row.loc[selected]
            converted_rank = len(selected) + 1 - ranks
            w = converted_rank / converted_rank.sum()
        else:  # equal
            w = pd.Series(1.0 / len(selected), index=selected)

        wgt.loc[date, selected] = w

    # Shift weights to next month
    wgt.index = wgt.index + BMonthBegin(1)

    # Backtest
    if start_date is None:
        start_date = wgt.index[0]
        weight_dates = wgt.index
    else:
        start_date = pd.to_datetime(start_date)
        valid_dates = wgt.index[wgt.index >= start_date]
        if len(valid_dates) == 0:
            return np.nan
        start_date = valid_dates[0]
        weight_dates = wgt[start_date:].index

    ret = price.pct_change()
    ret_sub = ret[start_date:].fillna(0.)
    all_dates = ret_sub.index

    if len(all_dates) == 0:
        return np.nan

    capital = 1000.0
    nav = pd.Series(index=all_dates, dtype=float)
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
                new_w = wgt.loc[date].values
                turn = np.sum(np.abs(new_w - prev_w)) / 2
                equity_after_cost = (prev_equity + dollar_pnl.sum()) * (1 - turn * tcost * 2)
            else:
                new_w = wgt.loc[date].values
                equity_after_cost = capital

            nav.loc[date] = equity_after_cost
            dollar_pos = new_w * equity_after_cost
            prev_w = new_w.copy()
        else:
            prev_date = all_dates[i - 1]
            prev_equity = nav.loc[prev_date]
            dollar_pnl = dollar_pos * ret_sub.loc[date].values
            dollar_pos += dollar_pnl
            nav.loc[date] = prev_equity + dollar_pnl.sum()
            prev_w = dollar_pos / dollar_pos.sum()

    # Calculate Sharpe
    nav = nav.dropna()
    if len(nav) < 2:
        return np.nan

    daily_ret = nav.pct_change().dropna()
    if len(daily_ret) == 0 or daily_ret.std() == 0:
        return np.nan

    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
    return sharpe


def optimize_sharpe(
    price: pd.DataFrame,
    top_k: int | None = 5,
    weight_method: str = "equal",
    tcost: float = 0.0,
    start_date: str | None = None,
    thresh: int = 10,
    n_samples: int | None = None,
    seed: int | None = None,
    progress_callback=None,
) -> dict:
    """
    Find optimal lookback windows and weights to maximize Sharpe ratio.

    Args:
        price: Daily price DataFrame
        top_k: Number of top-ranked assets (None for all)
        weight_method: Weighting method
        tcost: Transaction cost
        start_date: Backtest start date
        thresh: Minimum valid tickers
        n_samples: Number of random samples to test (None for full grid search)
        seed: Random seed for reproducibility (only used with n_samples)
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        Dict with optimal parameters and results
    """
    # Precompute momentums for all windows
    momentums = precompute_momentums(price)

    # Precompute volatility
    vol = calc_vol(price).resample('BM').last()

    # Generate all combinations
    lookback_combos = generate_lookback_combinations()
    weight_combos = generate_weight_combinations(step=10)

    # Create all parameter combinations
    all_combos = [
        (lb, wt) for lb in lookback_combos for wt in weight_combos
    ]
    total_possible = len(all_combos)

    # Random sampling if requested
    if n_samples is not None and n_samples < total_possible:
        if seed is not None:
            random.seed(seed)
        all_combos = random.sample(all_combos, n_samples)

    total = len(all_combos)

    best_sharpe = -np.inf
    best_params = None
    results = []

    for current, (lb, wt) in enumerate(all_combos, 1):
        short, mid, long = lb
        short_wgt, mid_wgt, long_wgt = wt

        if progress_callback:
            progress_callback(current, total)

        # Generate signal using precomputed momentums
        signal = combine_signals(
            momentums, short, mid, long,
            short_wgt, mid_wgt, long_wgt,
            thresh=thresh
        )

        if len(signal) == 0:
            continue

        # Run fast backtest
        sharpe = fast_backtest_sharpe(
            price, signal, vol,
            top_k=top_k,
            weight_method=weight_method,
            tcost=tcost,
            start_date=start_date,
        )

        if np.isnan(sharpe):
            continue

        results.append({
            'short_window': short,
            'mid_window': mid,
            'long_window': long,
            'short_wgt': short_wgt,
            'mid_wgt': mid_wgt,
            'long_wgt': long_wgt,
            'sharpe': sharpe,
        })

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = {
                'short_window': short,
                'mid_window': mid,
                'long_window': long,
                'short_wgt': short_wgt,
                'mid_wgt': mid_wgt,
                'long_wgt': long_wgt,
            }

    return {
        'best_params': best_params,
        'best_sharpe': best_sharpe,
        'total_combinations': total,
        'total_possible': total_possible,
        'valid_combinations': len(results),
        'all_results': results,
    }
