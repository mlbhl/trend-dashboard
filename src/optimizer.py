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
    allowed = [1,2,3,6,9,10,11,12]
    for short in allowed:
        for mid in allowed:
            if mid <= short:
                continue
            for long in allowed:
                if long <= mid:
                    continue
                combinations.append((short, mid, long))
    return combinations


def generate_weight_combinations(step: int = 10) -> list[tuple[float, float, float]]:
    """Generate all weight combinations summing to 100% with given step."""
    combinations = []
    values = list(range(10, 81, step))  # 10, 20, ..., 80
    for w1 in values:
        for w2 in values:
            w3 = 100 - w1 - w2
            if 10 <= w3 <= 80:
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


def optimize_sharpe_period(
    price: pd.DataFrame,
    momentums: dict[int, pd.DataFrame],
    vol: pd.DataFrame,
    top_k: int | None = 5,
    weight_method: str = "equal",
    tcost: float = 0.0,
    start_date: str | None = None,
    end_date: str | None = None,
    thresh: int = 10,
    n_samples: int | None = None,
    seed: int | None = None,
    progress_callback=None,
) -> dict:
    """
    Optimize parameters for a specific period using precomputed data.
    Internal function used by both optimize_sharpe and walk_forward_optimize.
    """
    # Filter price data to the period
    if start_date:
        price = price[start_date:]
    if end_date:
        price = price[:end_date]

    # Generate all combinations
    lookback_combos = generate_lookback_combinations()
    weight_combos = generate_weight_combinations(step=10)

    all_combos = [(lb, wt) for lb in lookback_combos for wt in weight_combos]
    total_possible = len(all_combos)

    if n_samples is not None and n_samples < total_possible:
        if seed is not None:
            random.seed(seed)
        all_combos = random.sample(all_combos, n_samples)

    total = len(all_combos)
    best_sharpe = -np.inf
    best_params = None

    for current, (lb, wt) in enumerate(all_combos, 1):
        short, mid, long = lb
        short_wgt, mid_wgt, long_wgt = wt

        if progress_callback:
            progress_callback(current, total)

        signal = combine_signals(
            momentums, short, mid, long,
            short_wgt, mid_wgt, long_wgt,
            thresh=thresh
        )

        if len(signal) == 0:
            continue

        sharpe = fast_backtest_sharpe(
            price, signal, vol,
            top_k=top_k,
            weight_method=weight_method,
            tcost=tcost,
            start_date=start_date,
        )

        if np.isnan(sharpe):
            continue

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
    }


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


def walk_forward_optimize(
    price: pd.DataFrame,
    train_months: int = 36,
    test_months: int = 12,
    step_months: int = 12,
    window_type: str = "rolling",
    top_k: int | None = 5,
    weight_method: str = "equal",
    tcost: float = 0.0,
    thresh: int = 10,
    n_samples: int | None = None,
    seed: int | None = None,
    bm_ticker: str | None = None,
    bm_data: pd.Series | None = None,
    progress_callback=None,
) -> dict:
    """
    Walk-forward optimization to reduce overfitting.

    Divides data into train/test periods:
    - Rolling: Fixed-size training window that moves forward
    - Expanding: Training window grows from start, always includes all past data

    Args:
        price: Daily price DataFrame
        train_months: Training period in months (minimum for expanding)
        test_months: Testing period in months
        step_months: Step size for rolling forward
        window_type: "rolling" (fixed window) or "expanding" (growing window)
        top_k: Number of top-ranked assets
        weight_method: Weighting method
        tcost: Transaction cost
        thresh: Minimum valid tickers
        n_samples: Number of random samples per fold
        seed: Random seed
        bm_ticker: Custom benchmark ticker name (None for equal weight)
        bm_data: Custom benchmark price data (required if bm_ticker is set)
        progress_callback: Optional callback(fold, total_folds, step_in_fold, total_steps)

    Returns:
        Dict with fold results, combined OOS performance, parameter stability
    """
    from dateutil.relativedelta import relativedelta

    # Precompute all data once
    momentums = precompute_momentums(price)
    vol = calc_vol(price).resample('BM').last()

    # Get date range
    start = price.index[0]
    end = price.index[-1]

    # Generate fold periods
    folds = []

    if window_type == "expanding":
        # Expanding window: train_start is always 'start', train_end grows
        # First fold starts after minimum train_months
        current_train_end = start + relativedelta(months=train_months)

        while True:
            test_start = current_train_end
            test_end = test_start + relativedelta(months=test_months)

            if test_end > end:
                break

            folds.append({
                'train_start': start,  # Always from the beginning
                'train_end': current_train_end,
                'test_start': test_start,
                'test_end': test_end,
            })

            current_train_end = current_train_end + relativedelta(months=step_months)
    else:
        # Rolling window: fixed-size training window
        current_train_start = start

        while True:
            train_end = current_train_start + relativedelta(months=train_months)
            test_start = train_end
            test_end = test_start + relativedelta(months=test_months)

            if test_end > end:
                break

            folds.append({
                'train_start': current_train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
            })

            current_train_start = current_train_start + relativedelta(months=step_months)

    if len(folds) == 0:
        return {
            'error': 'Not enough data for walk-forward optimization',
            'folds': [],
            'oos_sharpe': np.nan,
            'is_sharpe_avg': np.nan,
        }

    total_folds = len(folds)
    fold_results = []
    oos_navs = []
    oos_bm_navs = []

    for fold_idx, fold in enumerate(folds):
        fold_seed = (seed + fold_idx) if seed is not None else None

        def fold_progress(current, total):
            if progress_callback:
                progress_callback(fold_idx + 1, total_folds, current, total)

        # Optimize on training period
        opt_result = optimize_sharpe_period(
            price=price,
            momentums=momentums,
            vol=vol,
            top_k=top_k,
            weight_method=weight_method,
            tcost=tcost,
            start_date=str(fold['train_start'].date()),
            end_date=str(fold['train_end'].date()),
            thresh=thresh,
            n_samples=n_samples,
            seed=fold_seed,
            progress_callback=fold_progress,
        )

        if opt_result['best_params'] is None:
            continue

        bp = opt_result['best_params']
        is_sharpe = opt_result['best_sharpe']

        # Test on out-of-sample period with optimized params
        signal = combine_signals(
            momentums,
            bp['short_window'], bp['mid_window'], bp['long_window'],
            bp['short_wgt'], bp['mid_wgt'], bp['long_wgt'],
            thresh=thresh
        )

        # Get OOS NAV for this fold
        oos_nav = fast_backtest_nav(
            price, signal, vol,
            top_k=top_k,
            weight_method=weight_method,
            tcost=tcost,
            start_date=str(fold['test_start'].date()),
            end_date=str(fold['test_end'].date()),
        )

        # Get OOS Benchmark NAV for this fold
        if bm_ticker and bm_data is not None:
            # Custom ticker benchmark (buy and hold)
            oos_bm_nav = fast_single_ticker_nav(
                bm_data,
                start_date=str(fold['test_start'].date()),
                end_date=str(fold['test_end'].date()),
            )
        else:
            # Equal weight benchmark
            oos_bm_nav = fast_benchmark_nav(
                price,
                tcost=tcost,
                start_date=str(fold['test_start'].date()),
                end_date=str(fold['test_end'].date()),
            )

        if oos_nav is not None and len(oos_nav) > 0:
            oos_navs.append(oos_nav)
        if oos_bm_nav is not None and len(oos_bm_nav) > 0:
            oos_bm_navs.append(oos_bm_nav)

        oos_sharpe = _calc_sharpe_from_nav(oos_nav) if oos_nav is not None else np.nan

        fold_results.append({
            'fold': fold_idx + 1,
            'train_start': fold['train_start'].strftime('%Y-%m-%d'),
            'train_end': fold['train_end'].strftime('%Y-%m-%d'),
            'test_start': fold['test_start'].strftime('%Y-%m-%d'),
            'test_end': fold['test_end'].strftime('%Y-%m-%d'),
            'params': bp,
            'is_sharpe': is_sharpe,
            'oos_sharpe': oos_sharpe,
        })

    # Combine all OOS NAVs
    combined_oos_nav = _combine_navs(oos_navs) if oos_navs else None
    combined_oos_bm_nav = _combine_navs(oos_bm_navs) if oos_bm_navs else None
    combined_oos_sharpe = _calc_sharpe_from_nav(combined_oos_nav) if combined_oos_nav is not None else np.nan
    combined_oos_bm_sharpe = _calc_sharpe_from_nav(combined_oos_bm_nav) if combined_oos_bm_nav is not None else np.nan

    # Calculate parameter stability
    param_stability = _calc_param_stability(fold_results) if fold_results else {}

    # Average in-sample Sharpe
    is_sharpes = [f['is_sharpe'] for f in fold_results if not np.isnan(f['is_sharpe'])]
    is_sharpe_avg = np.mean(is_sharpes) if is_sharpes else np.nan

    # Average OOS Sharpe
    oos_sharpes = [f['oos_sharpe'] for f in fold_results if not np.isnan(f['oos_sharpe'])]
    oos_sharpe_avg = np.mean(oos_sharpes) if oos_sharpes else np.nan

    # Final training: train on all available data up to now
    # For rolling: use last train_months of data
    # For expanding: use all data from start
    if window_type == "expanding":
        final_train_start = start
    else:
        final_train_start = end - relativedelta(months=train_months)
        if final_train_start < start:
            final_train_start = start

    # Progress callback for final training (use fold index = total_folds + 1)
    def final_progress(current, total):
        if progress_callback:
            progress_callback(total_folds + 1, total_folds + 1, current, total)

    final_result = optimize_sharpe_period(
        price=price,
        momentums=momentums,
        vol=vol,
        top_k=top_k,
        weight_method=weight_method,
        tcost=tcost,
        start_date=str(final_train_start.date()) if hasattr(final_train_start, 'date') else str(final_train_start),
        end_date=str(end.date()) if hasattr(end, 'date') else str(end),
        thresh=thresh,
        n_samples=n_samples,
        seed=(seed + total_folds) if seed is not None else None,
        progress_callback=final_progress,
    )

    final_params = final_result['best_params']
    final_sharpe = final_result['best_sharpe']
    final_train_period = {
        'start': final_train_start.strftime('%Y-%m-%d') if hasattr(final_train_start, 'strftime') else str(final_train_start),
        'end': end.strftime('%Y-%m-%d') if hasattr(end, 'strftime') else str(end),
    }

    # Save OOS date range for dynamic benchmark calculation later
    oos_date_range = None
    if combined_oos_nav is not None and len(combined_oos_nav) > 0:
        oos_date_range = {
            'start': combined_oos_nav.index[0].strftime('%Y-%m-%d'),
            'end': combined_oos_nav.index[-1].strftime('%Y-%m-%d'),
        }

    return {
        'folds': fold_results,
        'combined_oos_nav': combined_oos_nav,
        'oos_date_range': oos_date_range,
        'oos_sharpe': combined_oos_sharpe,
        'oos_sharpe_avg': oos_sharpe_avg,
        'is_sharpe_avg': is_sharpe_avg,
        'sharpe_decay': is_sharpe_avg - oos_sharpe_avg if not np.isnan(is_sharpe_avg) and not np.isnan(oos_sharpe_avg) else np.nan,
        'param_stability': param_stability,
        'total_folds': total_folds,
        'valid_folds': len(fold_results),
        'window_type': window_type,
        'final_params': final_params,
        'final_sharpe': final_sharpe,
        'final_train_period': final_train_period,
    }


def fast_backtest_nav(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    vol: pd.DataFrame,
    top_k: int | None = 5,
    weight_method: str = "equal",
    tcost: float = 0.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series | None:
    """
    Fast backtest returning NAV series for a specific period.
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
        else:
            w = pd.Series(1.0 / len(selected), index=selected)

        wgt.loc[date, selected] = w

    wgt.index = wgt.index + BMonthBegin(1)

    # Determine date range
    if start_date is None:
        start_date = wgt.index[0]
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is not None:
        end_date = pd.to_datetime(end_date)

    valid_dates = wgt.index[wgt.index >= start_date]
    if end_date is not None:
        valid_dates = valid_dates[valid_dates <= end_date]

    if len(valid_dates) == 0:
        return None

    start_date = valid_dates[0]
    weight_dates = wgt[start_date:end_date].index if end_date else wgt[start_date:].index

    ret = price.pct_change()
    ret_sub = ret[start_date:end_date].fillna(0.) if end_date else ret[start_date:].fillna(0.)
    all_dates = ret_sub.index

    if len(all_dates) == 0:
        return None

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

    return nav.dropna()


def _calc_sharpe_from_nav(nav: pd.Series | None) -> float:
    """Calculate Sharpe ratio from NAV series."""
    if nav is None or len(nav) < 2:
        return np.nan
    daily_ret = nav.pct_change().dropna()
    if len(daily_ret) == 0 or daily_ret.std() == 0:
        return np.nan
    return daily_ret.mean() / daily_ret.std() * np.sqrt(252)


def _combine_navs(navs: list[pd.Series]) -> pd.Series | None:
    """Combine multiple NAV series by chaining returns."""
    if not navs:
        return None

    combined = navs[0].copy()
    for nav in navs[1:]:
        if len(nav) == 0:
            continue
        # Chain: scale next NAV to start from last combined value
        scale = combined.iloc[-1] / nav.iloc[0]
        scaled_nav = nav * scale
        # Avoid duplicate indices
        new_dates = scaled_nav.index[scaled_nav.index > combined.index[-1]]
        combined = pd.concat([combined, scaled_nav.loc[new_dates]])

    return combined


def _calc_param_stability(fold_results: list[dict]) -> dict:
    """Calculate parameter stability across folds."""
    if len(fold_results) < 2:
        return {}

    params_list = [f['params'] for f in fold_results]

    stability = {}
    for key in ['short_window', 'mid_window', 'long_window', 'short_wgt', 'mid_wgt', 'long_wgt']:
        values = [p[key] for p in params_list]
        stability[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }

    return stability


def fast_benchmark_nav(
    price: pd.DataFrame,
    tcost: float = 0.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series | None:
    """
    Compute equal-weight benchmark NAV for a specific period.
    All available tickers get equal weight at each rebalancing.
    """
    # Resample to monthly for rebalancing dates
    price_m = price.resample('BM').last()

    # Compute equal weights for all valid tickers at each month
    wgt = pd.DataFrame(0., index=price_m.index, columns=price_m.columns)
    for date in price_m.index:
        valid = price_m.loc[date].dropna().index
        if len(valid) > 0:
            wgt.loc[date, valid] = 1.0 / len(valid)

    # Shift weights to next month beginning
    wgt.index = wgt.index + BMonthBegin(1)

    # Determine date range
    if start_date is None:
        start_date = wgt.index[0]
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is not None:
        end_date = pd.to_datetime(end_date)

    valid_dates = wgt.index[wgt.index >= start_date]
    if end_date is not None:
        valid_dates = valid_dates[valid_dates <= end_date]

    if len(valid_dates) == 0:
        return None

    start_date = valid_dates[0]
    weight_dates = wgt[start_date:end_date].index if end_date else wgt[start_date:].index

    ret = price.pct_change()
    ret_sub = ret[start_date:end_date].fillna(0.) if end_date else ret[start_date:].fillna(0.)
    all_dates = ret_sub.index

    if len(all_dates) == 0:
        return None

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

    return nav.dropna()


def fast_single_ticker_nav(
    price_series: pd.Series,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series | None:
    """
    Compute NAV for a single ticker (buy and hold) for a specific period.
    """
    if start_date is None:
        start_date = price_series.index[0]
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is not None:
        end_date = pd.to_datetime(end_date)

    # Filter to date range
    if end_date:
        price_sub = price_series[start_date:end_date]
    else:
        price_sub = price_series[start_date:]

    if len(price_sub) == 0:
        return None

    # Rebase to 1000
    nav = price_sub / price_sub.iloc[0] * 1000

    return nav
