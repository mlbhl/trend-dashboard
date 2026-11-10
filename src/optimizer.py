"""Optimizer for momentum strategy parameters."""

import random
import numpy as np
import pandas as pd

from .portfolio import (
    backtest,
    blend_core_satellite,
    compute_weight_top_k,
    equal_weight_benchmark,
    rebalance_dates,
    to_rebalance_dates,
)
from .metrics import sharpe_from_nav
from .signals import calc_vol


def precompute_momentums(price: pd.DataFrame, max_window: int = 12) -> dict[int, pd.DataFrame]:
    """
    Precompute momentum ranks for all windows (1 to max_window).

    Returns:
        Dict mapping window -> momentum rank DataFrame
    """
    price_m = price.resample('BME').last()
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
    values = list(range(0, 101, step))  # 0, 10, ..., 100
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


def _with_core(sat_wgt: pd.DataFrame, price: pd.DataFrame, core_config: dict | None) -> pd.DataFrame:
    """Blend satellite weights with the core exactly as the dashboard does.

    Always returns weights on price's columns: backtest() pairs weights and returns by position.
    """
    if not core_config:
        return sat_wgt.reindex(columns=price.columns, fill_value=0.0)
    return blend_core_satellite(
        sat_wgt, price, core_config["tickers"], core_config["weights"], core_config["ratio"]
    )


def _strategy_weights(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    vol: pd.DataFrame,
    top_k: int | None,
    weight_method: str,
    core_config: dict | None,
) -> pd.DataFrame:
    """Dashboard top-k weights (+ core blend), moved to their rebalance dates."""
    wgt, _ = compute_weight_top_k(price, signal, top_k=top_k, weight_method=weight_method, vol=vol)
    return to_rebalance_dates(_with_core(wgt, price, core_config))


def _period_nav(
    price: pd.DataFrame,
    wgt: pd.DataFrame,
    tcost: float,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series | None:
    """portfolio.backtest() NAV on [start_date, end_date], as Run Analysis shows that window.

    The backtest starts at the last rebalance on or before start_date (the positions held on
    start_date; the first rebalance after it if there is none), then the NAV is cut to the window.
    Returns None when the window holds no position.
    """
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        price, wgt = price[:end_date], wgt[:end_date]
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        opened = wgt.index[wgt.index <= start_date]
        if len(opened):
            wgt = wgt[opened[-1]:]
    if len(wgt) == 0:
        return None
    price = price[wgt.index[0]:]
    if len(price) == 0:
        return None
    nav, _ = backtest(price, wgt, capital=1000.0, tcost=tcost)
    nav = nav.dropna()
    if start_date is not None:
        nav = nav[start_date:]
    return nav if len(nav) else None


def _holding_month_start(ts) -> pd.Timestamp:
    """Rebalance date that opens the holding month of `ts`: previous business month-end + lag.

    Walk-forward folds start and end on these dates, so consecutive test windows share an edge.
    """
    ts = pd.Timestamp(ts)
    prev_month_end = ts.replace(day=1) - pd.Timedelta(days=1)
    return rebalance_dates([pd.offsets.BMonthEnd().rollback(prev_month_end)])[0]


def _first_rebalance_on_or_after(ts) -> pd.Timestamp:
    """First rebalance date on or after `ts`."""
    ts = pd.Timestamp(ts)
    opening = _holding_month_start(ts)
    return opening if opening >= ts else _holding_month_start(ts + pd.offsets.MonthBegin(1))


def _benchmark_weights(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    core_config: dict | None = None,
) -> pd.DataFrame:
    """Dashboard benchmark weights at rebalance dates: equal weight across tickers with a signal (+ core)."""
    return to_rebalance_dates(_with_core(equal_weight_benchmark(signal), price, core_config))


def _strategy_nav(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    vol: pd.DataFrame,
    top_k: int | None = 5,
    weight_method: str = "equal",
    tcost: float = 0.0,
    start_date: str | None = None,
    end_date: str | None = None,
    core_config: dict | None = None,
) -> pd.Series | None:
    """Strategy NAV for a period, using the same weight and backtest code as the dashboard."""
    wgt = _strategy_weights(price, signal, vol, top_k, weight_method, core_config)
    return _period_nav(price, wgt, tcost, start_date, end_date)


def _strategy_sharpe(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    vol: pd.DataFrame,
    top_k: int | None = 5,
    weight_method: str = "equal",
    tcost: float = 0.0,
    start_date: str | None = None,
    end_date: str | None = None,
    core_config: dict | None = None,
) -> float:
    """Sharpe ratio of the strategy on [start_date, end_date], as Run Analysis shows it."""
    nav = _strategy_nav(price, signal, vol, top_k, weight_method, tcost, start_date, end_date, core_config)
    return sharpe_from_nav(nav)


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
    core_config: dict | None = None,
) -> dict:
    """
    Optimize parameters for a specific period using precomputed data.
    Internal function used by both optimize_sharpe and walk_forward_optimize.
    """
    # Nothing after the period may be used. (The start is handled by _period_nav: positions held
    # on start_date come from the rebalance before it.)
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

        signal = combine_signals(
            momentums, short, mid, long,
            short_wgt, mid_wgt, long_wgt,
            thresh=thresh
        )

        if len(signal) == 0:
            if progress_callback:
                progress_callback(current, total)
            continue

        sharpe = _strategy_sharpe(
            price, signal, vol,
            top_k=top_k,
            weight_method=weight_method,
            tcost=tcost,
            start_date=start_date,
            end_date=end_date,
            core_config=core_config,
        )

        if np.isnan(sharpe):
            if progress_callback:
                progress_callback(current, total)
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

        if progress_callback:
            progress_callback(current, total)

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
    end_date: str | None = None,
    thresh: int = 10,
    n_samples: int | None = None,
    seed: int | None = None,
    progress_callback=None,
    core_config: dict | None = None,
    universe: list[str] | None = None,
) -> dict:
    """
    Find optimal lookback windows and weights to maximize Sharpe ratio.

    The score is the Sharpe Run Analysis shows for [start_date, end_date] with those parameters.

    Args:
        price: Daily price DataFrame
        top_k: Number of top-ranked assets (None for all)
        weight_method: Weighting method
        tcost: Transaction cost
        start_date: Backtest start date
        end_date: Backtest end date (data after it is not used)
        thresh: Minimum valid tickers
        n_samples: Number of random samples to test (None for full grid search)
        seed: Random seed for reproducibility (only used with n_samples)
        progress_callback: Optional callback(current, total) for progress updates
        universe: Tickers the strategy may select (satellite universe). Defaults to all price
            columns. In core-satellite mode, price also carries core tickers that are not
            selectable; they are used only for the core weights.

    Returns:
        Dict with optimal parameters and results
    """
    # Signals and volatility come only from the selectable universe (as in the dashboard)
    sat_price = price if universe is None else price[[t for t in universe if t in price.columns]]
    momentums = precompute_momentums(sat_price)
    vol = calc_vol(sat_price).resample('BME').last()

    return optimize_sharpe_period(
        price=price,
        momentums=momentums,
        vol=vol,
        top_k=top_k,
        weight_method=weight_method,
        tcost=tcost,
        start_date=start_date,
        end_date=end_date,
        thresh=thresh,
        n_samples=n_samples,
        seed=seed,
        progress_callback=progress_callback,
        core_config=core_config,
    )


def walk_forward_optimize(
    price: pd.DataFrame,
    train_months: int = 36,
    test_months: int = 12,
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
    core_config: dict | None = None,
    universe: list[str] | None = None,
) -> dict:
    """
    Walk-forward optimization to reduce overfitting.

    Divides data into train/test periods:
    - Rolling: Fixed-size training window that moves forward
    - Expanding: Training window grows from start, always includes all past data

    Args:
        price: Daily price DataFrame
        train_months: Training period in months (minimum for expanding)
        test_months: Testing period in months; folds also step forward by this much, so the
            out-of-sample windows are back to back (a different step would overlap or leave gaps)
        window_type: "rolling" (fixed window) or "expanding" (growing window)
        top_k: Number of top-ranked assets
        weight_method: Weighting method
        tcost: Transaction cost
        thresh: Minimum valid tickers
        n_samples: Number of random samples per fold
        seed: Random seed
        bm_ticker: Custom benchmark label; when set, bm_data (a price series) is the OOS benchmark
                   instead of the equal-weight one
        bm_data: Custom benchmark price data (required if bm_ticker is set)
        progress_callback: Optional callback(fold, total_folds, step_in_fold, total_steps)
        universe: Tickers the strategy may select (satellite universe). Defaults to all price
            columns. In core-satellite mode, price also carries core tickers that are not
            selectable; they are used only for the core weights.

    Returns:
        Dict with fold results, combined OOS performance, parameter stability
    """
    from dateutil.relativedelta import relativedelta

    # Precompute all data once — signals/vol only from the selectable universe (as in the dashboard)
    sat_price = price if universe is None else price[[t for t in universe if t in price.columns]]
    momentums = precompute_momentums(sat_price)
    vol = calc_vol(sat_price).resample('BME').last()

    # Get date range
    start = price.index[0]
    end = price.index[-1]

    # Fold periods. Boundaries are snapped to rebalance dates (the date that opens a holding
    # month), so fold k's test window ends exactly where fold k+1's begins: no gap, no overlap.
    if test_months < 1:
        raise ValueError("test_months must be at least 1")
    folds = []
    first_train_end = start + relativedelta(months=train_months)
    k = 0
    while True:
        train_end_cal = first_train_end + relativedelta(months=k * test_months)
        test_start = _holding_month_start(train_end_cal)
        test_end = _holding_month_start(train_end_cal + relativedelta(months=test_months))
        if test_end > end:
            break
        folds.append({
            # expanding: always from the beginning; rolling: fixed-size window moving forward
            'train_start': start if window_type == "expanding" else start + relativedelta(months=k * test_months),
            'train_end': test_start,
            'test_start': test_start,
            'test_end': test_end,
        })
        k += 1

    if len(folds) == 0:
        return {
            'error': 'Not enough data for walk-forward optimization',
            'folds': [],
            'oos_sharpe': np.nan,
            'is_sharpe_avg': np.nan,
        }

    total_folds = len(folds)
    fold_results = []
    strat_parts, bm_parts = [], []   # each fold's weights inside its own test window
    oos_start = oos_end = None
    last_params = None

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
            # training is scored from its first rebalance (a fresh start, not an inherited position)
            start_date=str(_first_rebalance_on_or_after(fold['train_start']).date()),
            end_date=str(fold['train_end'].date()),
            thresh=thresh,
            n_samples=n_samples,
            seed=fold_seed,
            progress_callback=fold_progress,
            core_config=core_config,
        )

        bp = opt_result['best_params']
        if bp is None:
            if last_params is None:
                continue  # out-of-sample period has not started yet
            bp = last_params  # re-optimisation failed: keep trading the previous parameters
        else:
            fold_results.append({
                'fold': fold_idx + 1,
                'train_start': fold['train_start'].strftime('%Y-%m-%d'),
                'train_end': fold['train_end'].strftime('%Y-%m-%d'),
                'test_start': fold['test_start'].strftime('%Y-%m-%d'),
                'test_end': fold['test_end'].strftime('%Y-%m-%d'),
                'params': bp,
                'is_sharpe': opt_result['best_sharpe'],
            })
        last_params = bp

        # Out-of-sample: this fold's parameters drive the rebalances inside its test window
        signal = combine_signals(
            momentums,
            bp['short_window'], bp['mid_window'], bp['long_window'],
            bp['short_wgt'], bp['mid_wgt'], bp['long_wgt'],
            thresh=thresh
        )
        wgt = _strategy_weights(price, signal, vol, top_k, weight_method, core_config)
        bm_wgt = _benchmark_weights(price, signal, core_config)
        in_window = (wgt.index >= fold['test_start']) & (wgt.index < fold['test_end'])
        strat_parts.append(wgt[in_window])
        bm_parts.append(bm_wgt[(bm_wgt.index >= fold['test_start']) & (bm_wgt.index < fold['test_end'])])
        oos_start = fold['test_start'] if oos_start is None else oos_start
        oos_end = fold['test_end']

    # One continuous out-of-sample backtest across all folds: the switch between folds is an
    # ordinary rebalance (turnover cost included), exactly like periodically re-optimising live.
    combined_oos_nav = combined_oos_bm_nav = None
    if strat_parts:
        combined_oos_nav = _period_nav(price, pd.concat(strat_parts), tcost, oos_start, oos_end)
        if bm_ticker and bm_data is not None:
            # Custom benchmark price series, rebased over the same window
            combined_oos_bm_nav = _single_ticker_nav(
                bm_data, start_date=str(oos_start.date()), end_date=str(oos_end.date())
            )
        else:
            # Equal weight across tickers with a signal, as in the dashboard (no tcost for BM)
            combined_oos_bm_nav = _period_nav(price, pd.concat(bm_parts), 0.0, oos_start, oos_end)
    combined_oos_sharpe = sharpe_from_nav(combined_oos_nav)
    for f in fold_results:
        segment = combined_oos_nav[f['test_start']:f['test_end']] if combined_oos_nav is not None else None
        f['oos_sharpe'] = sharpe_from_nav(segment)

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
            progress_callback(total_folds + 1, total_folds, current, total)

    final_result = optimize_sharpe_period(
        price=price,
        momentums=momentums,
        vol=vol,
        top_k=top_k,
        weight_method=weight_method,
        tcost=tcost,
        start_date=str(_first_rebalance_on_or_after(final_train_start).date()),
        end_date=str(end.date()) if hasattr(end, 'date') else str(end),
        thresh=thresh,
        n_samples=n_samples,
        seed=(seed + total_folds) if seed is not None else None,
        progress_callback=final_progress,
        core_config=core_config,
    )

    final_params = final_result['best_params']
    final_sharpe = final_result['best_sharpe']
    final_train_period = {
        'start': final_train_start.strftime('%Y-%m-%d') if hasattr(final_train_start, 'strftime') else str(final_train_start),
        'end': end.strftime('%Y-%m-%d') if hasattr(end, 'strftime') else str(end),
    }

    return {
        'folds': fold_results,
        'combined_oos_nav': combined_oos_nav,
        'combined_oos_bm_nav': combined_oos_bm_nav,
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


def _single_ticker_nav(
    price_series: pd.Series,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series | None:
    """Benchmark price series (custom BM) rebased to 1000 over a period."""
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
