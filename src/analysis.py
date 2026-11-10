"""Run Analysis pipeline, shared by the dashboard and scripts/generate_report.py.

Pure pandas (no Dash): errors are raised as AnalysisError and warnings returned as strings,
so the dashboard callback only wraps them into alerts and dcc.Store data.
"""

from dataclasses import dataclass, field

import pandas as pd

from .config import (
    DEFAULT_BACKTEST_START_DATE,
    DEFAULT_BM_TYPE,
    DEFAULT_CORE_WEIGHT_PCT,
    DEFAULT_LONG_WEIGHT,
    DEFAULT_LONG_WINDOW,
    DEFAULT_MID_WEIGHT,
    DEFAULT_MID_WINDOW,
    DEFAULT_N_QUANTILES,
    DEFAULT_SHORT_WEIGHT,
    DEFAULT_SHORT_WINDOW,
    DEFAULT_START_DATE,
    DEFAULT_STRATEGY_MODE,
    DEFAULT_STRATEGY_TYPE,
    DEFAULT_TCOST_PCT,
    DEFAULT_THRESH,
    DEFAULT_TOP_K,
    DEFAULT_WEIGHT_METHOD,
    DEFAULT_WEIGHT_MODE,
)
from .data import load_price_data
from .portfolio import (
    backtest,
    blend_core_satellite,
    compute_weight_core,
    compute_weight_quantile,
    compute_weight_top_k,
    equal_weight_benchmark,
    rebalance_dates,
    run_quantile_backtest,
    run_top_k_backtest,
    to_rebalance_dates,
)
from .signals import generate_signal


class AnalysisError(Exception):
    """Analysis cannot run with the given inputs; the message is shown to the user."""


@dataclass
class AnalysisResult:
    params: dict                         # settings + display window, as stored for the charts
    navs: pd.DataFrame                   # display window: strategy column(s) + BM column
    signal: pd.DataFrame                 # full signal, incl. the tentative current-month row
    weights: dict[str, pd.DataFrame]     # holdings at rebalance dates, cut at display end
    strat_name: str                      # "Top-5", "All" or "" for quantile mode
    warnings: list[str] = field(default_factory=list)


def check_whole_number(value, minimum: int, label: str) -> str | None:
    """Problem with a numeric sidebar input, or None when it is a whole number >= minimum.

    Blank is a problem too: inputs are never silently replaced by a default. Number inputs also
    report a value outside their min/max as blank, so the message covers both cases.
    """
    if value is None or value == "":
        return f"{label}: enter a whole number of at least {minimum}."
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{label} must be a number."
    if number != int(number):
        return f"{label} must be a whole number."
    if number < minimum:
        return f"{label} must be at least {minimum}."
    return None


def check_tcost(value) -> str | None:
    """Problem with the transaction cost input (percent, 0 to 1), or None.

    The number input sends None when it is blank, out of range or off its 0.01 step:
    that must not run as zero cost.
    """
    problem = "Transaction cost: enter a number from 0 to 1 (%)."
    if value is None or value == "":
        return problem
    try:
        number = float(value)
    except (TypeError, ValueError):
        return problem
    return None if 0 <= number <= 1 else problem


def check_backtest_dates(start, end) -> str | None:
    """Problem with the Backtest Start / End Date pair, or None (no end date = up to the latest data)."""
    if end is not None and pd.to_datetime(end) <= pd.to_datetime(start):
        return "Backtest End Date must be after Backtest Start Date."
    return None


def parse_seed(value) -> int | None:
    """Optimizer seed: blank = random (None), otherwise a whole number >= 0 (AnalysisError if not)."""
    if value is None or str(value).strip() == "":
        return None
    problem = check_whole_number(str(value).strip(), 0, "Seed")
    if problem:
        raise AnalysisError(problem)
    return int(float(value))


def parse_core_config(core_config: list | None, core_weight_pct) -> dict:
    """Validated core-satellite settings: {"tickers", "weights", "ratio"}.

    Rows without a ticker or with a non-positive weight are ignored; rows of the same ticker add up
    (as in the custom benchmark). Raises AnalysisError when no core ticker is left or the core
    allocation is not between 1 and 99 %.
    """
    merged: dict[str, float] = {}
    for row in core_config or []:
        ticker = (row.get("ticker") or "").strip().upper()
        weight = float(row.get("weight") or 0)
        if ticker and weight > 0:
            merged[ticker] = merged.get(ticker, 0.0) + weight
    tickers, weights = list(merged), list(merged.values())
    if not tickers:
        raise AnalysisError(
            "Core-Satellite mode requires core tickers. Please configure and Apply core settings."
        )
    if core_weight_pct is None or core_weight_pct == "" or not 1 <= float(core_weight_pct) <= 99:
        raise AnalysisError("Core allocation must be between 1 and 99%.")
    return {"tickers": tickers, "weights": weights, "ratio": float(core_weight_pct) / 100.0}


def with_core_prices(dataset: pd.DataFrame, core_tickers: list[str], start_date: str) -> tuple:
    """(prices of the selected universe + core tickers outside it, core tickers without data)."""
    extra = [t for t in core_tickers if t not in dataset.columns]
    if not extra:
        return dataset, []
    expanded, missing = load_price_data(list(dataset.columns) + extra, start_date=start_date)
    return expanded, [t for t in missing if t in core_tickers]


def build_composite_bm(bm_config: list, start_date: str) -> tuple:
    """Build a composite benchmark from multiple tickers with weights.

    Args:
        bm_config: List of dicts [{"ticker": "SPY", "weight": 60}, ...]
        start_date: Start date for price data download.

    Returns:
        (bm_series, bm_label, warnings) where bm_series is a pd.Series of
        composite price, bm_label is a display string, and warnings is a list
        of warning strings for missing tickers.
    """
    if not bm_config:
        return None, None, []

    tickers = [row["ticker"].strip().upper() for row in bm_config if row.get("ticker")]
    weights = [float(row.get("weight", 0)) for row in bm_config if row.get("ticker")]

    if not tickers:
        return None, None, []

    # Convert percentage weights to decimals (e.g. 60 → 0.6)
    # No normalization: <100% implies cash, >100% implies leverage
    dec_weights = [w / 100.0 for w in weights]

    # Load all ticker data at once
    all_data, missing = load_price_data(tickers, start_date=start_date)

    warn_msgs = []
    if missing:
        warn_msgs.append(f"BM ticker(s) not found: {', '.join(missing)}")

    # Filter to found tickers and their weights
    found = []
    found_weights = []
    for t, w in zip(tickers, dec_weights):
        if t in all_data.columns:
            found.append(t)
            found_weights.append(w)
    if not found:
        return None, None, warn_msgs

    prices = all_data[found]
    daily_ret = prices.pct_change().dropna()
    if daily_ret.empty:
        return None, None, warn_msgs

    # Weighted daily return → cumulative price series
    composite_ret = daily_ret.multiply(found_weights, axis=1).sum(axis=1)
    composite_price = (1 + composite_ret).cumprod()
    composite_price.iloc[0] = 1.0  # set initial value

    # Build label from original config (only found tickers with original weights)
    label_parts = []
    for t, w in zip(tickers, weights):
        if t in all_data.columns:
            label_parts.append(f"{t} {w:.1f}%")
    bm_label = " + ".join(label_parts)

    return composite_price, bm_label, warn_msgs


def _check_quantiles_filled(signal: pd.DataFrame, n_quantiles: int, display_start, display_end) -> None:
    """Every quantile must hold a ticker in every month shown (n ranked tickers >= n quantiles).

    Months before the shown window may have fewer ranked tickers: their empty buckets are cash
    and do not affect the shown returns.
    """
    n_ranked = signal.notna().sum(axis=1)
    n_ranked.index = rebalance_dates(n_ranked.index)
    shown = n_ranked[:display_end]
    opened = shown.index[shown.index <= display_start]  # the rebalance holding on display_start
    if len(opened):
        shown = shown[opened[-1]:]
    if len(shown) and shown.min() < n_quantiles:
        fewest = int(shown.min())
        raise AnalysisError(
            f"Number of Quantiles ({n_quantiles}) is more than the tickers with a signal in some months "
            f"(as few as {fewest}), so some quantiles would hold nothing. Use {fewest} or fewer."
        )


def run_analysis(
    tickers: list[str],
    start_date: str = DEFAULT_START_DATE,
    backtest_start_date: str = DEFAULT_BACKTEST_START_DATE,
    backtest_end_date: str | None = None,
    thresh: int = DEFAULT_THRESH,
    short_window: int = DEFAULT_SHORT_WINDOW,
    mid_window: int = DEFAULT_MID_WINDOW,
    long_window: int = DEFAULT_LONG_WINDOW,
    weight_mode: str = DEFAULT_WEIGHT_MODE,
    short_wgt: float = DEFAULT_SHORT_WEIGHT,
    mid_wgt: float = DEFAULT_MID_WEIGHT,
    long_wgt: float = DEFAULT_LONG_WEIGHT,
    strategy_type: str = DEFAULT_STRATEGY_TYPE,
    select_all: bool = False,
    top_k: int | None = DEFAULT_TOP_K,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    weight_method: str = DEFAULT_WEIGHT_METHOD,
    tcost_pct: float | None = DEFAULT_TCOST_PCT,
    bm_type: str = DEFAULT_BM_TYPE,
    bm_config: list | None = None,
    strategy_mode: str = DEFAULT_STRATEGY_MODE,
    core_weight_pct: float | None = DEFAULT_CORE_WEIGHT_PCT,
    core_config: list | None = None,
) -> AnalysisResult:
    """Everything the Run Analysis button computes. Arguments mirror the sidebar controls;
    defaults are the sidebar defaults (src/config.py). tcost_pct is one-way, in percent."""
    # Core-Satellite applies to Top-K only: quantile buckets are a signal diagnostic, and a shared
    # core would wash out the spread between them (and the benchmark would no longer be comparable).
    notices = []
    if strategy_mode == "core-satellite" and strategy_type != "Top-K":
        strategy_mode = "single"
        notices.append("Core-Satellite applies to Top-K only. Quantile results exclude the core.")

    # Validate settings (raise with a user-facing message; never fall back to a default)
    core = parse_core_config(core_config, core_weight_pct) if strategy_mode == "core-satellite" else None
    for problem in (check_backtest_dates(backtest_start_date, backtest_end_date), check_tcost(tcost_pct)):
        if problem:
            raise AnalysisError(problem)

    # Calculate weights
    if weight_mode == "equal":
        short_wgt = mid_wgt = long_wgt = 1 / 3
    else:
        total = short_wgt + mid_wgt + long_wgt
        if total <= 0:  # every rank would tie: the first tickers in column order would be held
            raise AnalysisError("Custom window weights: set at least one of Short / Mid / Long above 0.")
        short_wgt, mid_wgt, long_wgt = short_wgt / total, mid_wgt / total, long_wgt / total

    # Handle top_k
    if select_all:
        top_k = None

    tcost = float(tcost_pct) / 100.0  # input is one-way, in percent

    # Load data
    dataset, missing = load_price_data(tickers, start_date=start_date)

    # Check if we have enough valid tickers
    if len(dataset.columns) < 2:
        error_msg = "Not enough valid tickers. Need at least 2 tickers with data."
        if missing:
            error_msg += f" Missing: {', '.join(missing)}"
        raise AnalysisError(error_msg)
    if len(dataset.columns) < thresh:
        raise AnalysisError(
            f"Only {len(dataset.columns)} tickers have price data, fewer than Min Valid Tickers ({thresh}). "
            "Add tickers or lower the threshold."
        )

    # Collect warnings
    warnings = list(notices)
    if missing:
        warnings.append(f"Data not found for: {', '.join(missing)}")

    # Load benchmark data if custom
    bm_data = None
    custom_bm_label = None
    if bm_type == "custom" and bm_config:
        bm_series, bm_label, bm_warns = build_composite_bm(bm_config, start_date)
        warnings.extend(bm_warns)
        if bm_series is not None:
            bm_data = bm_series
            custom_bm_label = bm_label
        else:
            warnings.append("Custom BM could not be built. Using Equal Weight BM instead.")
    elif bm_type == "custom":
        warnings.append("Custom BM has no tickers applied. Using Equal Weight BM instead.")

    # Generate signal
    signal = generate_signal(
        dataset,
        short_window=short_window,
        mid_window=mid_window,
        long_window=long_window,
        short_wgt=short_wgt,
        mid_wgt=mid_wgt,
        long_wgt=long_wgt,
        thresh=thresh,
    )
    if signal.empty:
        raise AnalysisError(
            "Not enough price history for the selected lookback windows and Min Valid Tickers."
        )

    # Core-satellite: core weights come from parse_core_config(); core-only tickers get prices too
    is_core_satellite = core is not None
    core_ratio = core["ratio"] if core else 0.0
    core_tickers_parsed = core["tickers"] if core else []
    core_weights_parsed = core["weights"] if core else []
    dataset_expanded = dataset
    if core:
        dataset_expanded, missing_core = with_core_prices(dataset, core_tickers_parsed, start_date)
        if missing_core:
            warnings.append(f"Core ticker(s) not found: {', '.join(missing_core)}")

    # Run backtest
    strat_name = ""
    if strategy_type == "Top-K":
        strat_name = "All" if top_k is None else f"Top-{top_k}"

        # Satellite weights (also shown in the holdings heatmap)
        sat_wgt, _ = compute_weight_top_k(dataset, signal, top_k=top_k, weight_method=weight_method)

        if is_core_satellite and core_tickers_parsed:
            # Core-only and satellite-only portfolios on the expanded dataset
            core_wgt = compute_weight_core(
                dataset_expanded, signal, core_tickers_parsed, core_weights_parsed
            )
            sat_wgt_expanded = sat_wgt.reindex(columns=dataset_expanded.columns, fill_value=0.0)

            # Blend weights (same function the optimizer uses)
            combined_wgt = blend_core_satellite(
                sat_wgt, dataset_expanded, core_tickers_parsed, core_weights_parsed, core_ratio
            )

            # Signal-date weights -> rebalance dates (config.IMPLEMENTATION_LAG)
            combined_nav, _ = backtest(dataset_expanded, to_rebalance_dates(combined_wgt), tcost=tcost)
            core_nav_only, _ = backtest(dataset_expanded, to_rebalance_dates(core_wgt), tcost=tcost)
            sat_nav_only, _ = backtest(dataset_expanded, to_rebalance_dates(sat_wgt_expanded), tcost=tcost)

            # Build NAV DataFrame with detailed core label
            total_core_w = sum(core_weights_parsed)
            core_label = " + ".join(
                f"{t} {w/total_core_w*100:.1f}%" for t, w in zip(core_tickers_parsed, core_weights_parsed)
            )
            core_pct = int(round(core_ratio * 100))
            sat_pct = 100 - core_pct
            navs = pd.DataFrame({
                f"Combined (Core {core_pct}% / Satellite {sat_pct}%)": combined_nav,
                f"Core ({core_label})": core_nav_only,
                f"Satellite ({strat_name})": sat_nav_only,
            })
        else:
            strat_nav, universe_bm_nav, _ = run_top_k_backtest(
                dataset,
                signal,
                top_k=top_k,
                weight_method=weight_method,
                tcost=tcost,
            )
            navs = pd.DataFrame({strat_name: strat_nav})
        weights = {strat_name: to_rebalance_dates(sat_wgt)}
    else:
        q_nav, _, universe_bm_nav = run_quantile_backtest(
            dataset,
            signal,
            n_quantiles=n_quantiles,
            tcost=tcost,
            weight_method=weight_method,
        )
        navs = q_nav.copy()

        # Weights for all quantiles
        q_wgts, _ = compute_weight_quantile(
            dataset, signal, n_quantiles=n_quantiles, weight_method=weight_method
        )
        weights = {q: to_rebalance_dates(wgt) for q, wgt in q_wgts.items()}

    # Handle benchmark
    if custom_bm_label and bm_data is not None:
        bm_col_name = f"BM ({custom_bm_label})"
        bm_aligned = bm_data.reindex(navs.index).ffill().bfill()
        navs[bm_col_name] = bm_aligned / bm_aligned.iloc[0] * 1000
    elif is_core_satellite and core_tickers_parsed:
        # Core-satellite BM: same core/satellite split but satellite = equal weight
        combined_bm_wgt = blend_core_satellite(
            equal_weight_benchmark(signal), dataset_expanded,
            core_tickers_parsed, core_weights_parsed, core_ratio,
        )
        combined_bm_nav, _ = backtest(dataset_expanded, to_rebalance_dates(combined_bm_wgt), tcost=0)
        bm_col_name = f"BM (Core {core_ratio:.0%} + EW {1-core_ratio:.0%})"
        navs[bm_col_name] = combined_bm_nav
    else:
        bm_col_name = "BM (Equal Weight)"
        navs[bm_col_name] = universe_bm_nav

    # Display window: backtest start / end dates
    backtest_start = pd.to_datetime(backtest_start_date)
    actual_start = navs.index[navs.index >= backtest_start]
    if len(actual_start) == 0:
        raise AnalysisError(f"No results from Backtest Start Date on: the data ends on {navs.index[-1]:%Y-%m-%d}.")
    display_start = actual_start[0]

    display_end = navs.index[-1]
    if backtest_end_date is not None:
        actual_end = navs.index[(navs.index >= display_start) & (navs.index <= pd.to_datetime(backtest_end_date))]
        if len(actual_end) < 2:
            raise AnalysisError(
                f"No results up to Backtest End Date: the backtest starts on {display_start:%Y-%m-%d}."
            )
        display_end = actual_end[-1]

    if strategy_type == "Quantile":
        _check_quantiles_filled(signal, n_quantiles, display_start, display_end)

    navs_display = navs.loc[display_start:display_end].copy()
    # Keep the full signal (incl. tentative BME row using latest close) for the Current Signal panel.
    # Holdings use rebalance-date weights cut at display_end, so only confirmed months show there.
    weights = {q: w.sort_index().loc[:display_end] for q, w in weights.items()}

    params = {
        "thresh": thresh,
        "short_window": short_window,
        "mid_window": mid_window,
        "long_window": long_window,
        "short_wgt": short_wgt,
        "mid_wgt": mid_wgt,
        "long_wgt": long_wgt,
        "top_k": top_k,
        "n_quantiles": n_quantiles,
        "weight_method": weight_method,
        "tcost": tcost,
        "strategy_type": strategy_type,
        "backtest_start_date": str(backtest_start_date),
        "backtest_end_date": str(backtest_end_date),
        "strategy_mode": strategy_mode,
        "core_ratio": core_ratio if is_core_satellite else None,
        "core_tickers": core_tickers_parsed if is_core_satellite else None,
        "bm_col_name": bm_col_name,
        "custom_bm_label": custom_bm_label if bm_type == "custom" else None,
        "display_start": display_start.strftime("%Y-%m-%d"),
        "display_end": navs_display.index[-1].strftime("%Y-%m-%d"),
        "data_end": dataset.index[-1].strftime("%Y-%m-%d"),
    }

    return AnalysisResult(
        params=params,
        navs=navs_display,
        signal=signal.copy(),
        weights=weights,
        strat_name=strat_name,
        warnings=warnings,
    )
