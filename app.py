"""Trend Dashboard - Momentum Strategy Analysis App"""

import streamlit as st
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BMonthBegin

from src.config import (
    ALPHA_LIST,
    TICKER_DESCRIPTIONS,
    DEFAULT_START_DATE,
    DEFAULT_BACKTEST_START_DATE,
    DEFAULT_SHORT_WINDOW,
    DEFAULT_MID_WINDOW,
    DEFAULT_LONG_WINDOW,
    DEFAULT_SHORT_WEIGHT,
    DEFAULT_MID_WEIGHT,
    DEFAULT_LONG_WEIGHT,
    DEFAULT_TOP_K,
    DEFAULT_TCOST,
    DEFAULT_THRESH,
    WEIGHT_METHODS,
)
from src.data import load_price_data
from src.signals import generate_signal, get_signal_ranking
from src.optimizer import optimize_sharpe, walk_forward_optimize
from src.portfolio import (
    compute_weight_top_k,
    run_quantile_backtest,
    run_top_k_backtest,
)
from src.metrics import summary_stats, rebase, calculate_kpis
from src.charts import (
    create_nav_chart,
    create_drawdown_chart,
    create_holding_heatmap,
    create_signal_category_table,
    create_quantile_spread_chart,
    create_returns_table,
)

# Page config
st.set_page_config(
    page_title="Trend Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("Momentum Strategy Dashboard")

# =============================================================================
# Sidebar - Parameters
# =============================================================================
st.sidebar.header("Parameters")

# Data settings
st.sidebar.subheader("Data Settings")
start_date = st.sidebar.date_input(
    "Download Start Date",
    value=pd.to_datetime(DEFAULT_START_DATE),
    min_value=pd.to_datetime("1990-01-01"),
    max_value=pd.to_datetime("today"),
    help="Data download period start"
)
backtest_start_date = st.sidebar.date_input(
    "Backtest Start Date",
    value=pd.to_datetime(DEFAULT_BACKTEST_START_DATE),
    min_value=pd.to_datetime("1990-01-01"),
    max_value=pd.to_datetime("today"),
    help="If data is insufficient, the earliest available date will be used."
)

# Proxy setting (optional)
# use_proxy = st.sidebar.checkbox("Use Proxy", value=False)
# proxy_url = None
# if use_proxy:
#     proxy_url = st.sidebar.text_input(
#         "Proxy URL",
#         placeholder="http://proxy.example.com:8080"
#     )

# Ticker selection
st.sidebar.subheader("Ticker Selection")

# Initialize custom tickers in session state
if 'custom_tickers' not in st.session_state:
    st.session_state['custom_tickers'] = []

# Clean up if optimization was stopped mid-execution
if st.session_state.get('opt_running', False):
    st.session_state['opt_running'] = False
    if 'optimization_result' in st.session_state:
        del st.session_state['optimization_result']

# Clean up if walk-forward was stopped mid-execution
if st.session_state.get('wf_running', False):
    st.session_state['wf_running'] = False
    if 'walk_forward_result' in st.session_state:
        del st.session_state['walk_forward_result']

# Apply optimized params if available (must be before widget creation)
if 'optimized_params' in st.session_state:
    bp = st.session_state['optimized_params']
    st.session_state['short_window'] = bp['short_window']
    st.session_state['mid_window'] = bp['mid_window']
    st.session_state['long_window'] = bp['long_window']
    st.session_state['short_wgt'] = bp['short_wgt']
    st.session_state['mid_wgt'] = bp['mid_wgt']
    st.session_state['long_wgt'] = bp['long_wgt']
    st.session_state['weight_mode'] = "Custom"
    del st.session_state['optimized_params']

# Initialize signal parameters in session state
if 'short_window' not in st.session_state:
    st.session_state['short_window'] = DEFAULT_SHORT_WINDOW
if 'mid_window' not in st.session_state:
    st.session_state['mid_window'] = DEFAULT_MID_WINDOW
if 'long_window' not in st.session_state:
    st.session_state['long_window'] = DEFAULT_LONG_WINDOW
if 'short_wgt' not in st.session_state:
    st.session_state['short_wgt'] = DEFAULT_SHORT_WEIGHT
if 'mid_wgt' not in st.session_state:
    st.session_state['mid_wgt'] = DEFAULT_MID_WEIGHT
if 'long_wgt' not in st.session_state:
    st.session_state['long_wgt'] = DEFAULT_LONG_WEIGHT
if 'weight_mode' not in st.session_state:
    st.session_state['weight_mode'] = "Custom"

# Combine default and custom tickers
available_tickers = ALPHA_LIST.copy() + st.session_state['custom_tickers']

# Initialize selected tickers (only on first load)
if 'ticker_select' not in st.session_state:
    st.session_state['ticker_select'] = available_tickers.copy()

# Callbacks (executed before widgets render)
def reset_tickers():
    st.session_state['ticker_select'] = ALPHA_LIST.copy()
    st.session_state['custom_tickers'] = []
    st.session_state['new_ticker_input'] = ""

def add_new_ticker():
    new_val = st.session_state.get('new_ticker_input', '').strip().upper()
    if new_val and new_val not in available_tickers:
        st.session_state['custom_tickers'].append(new_val)
        current = list(st.session_state.get('ticker_select', []))
        current.append(new_val)
        st.session_state['ticker_select'] = current
    st.session_state['new_ticker_input'] = ""

# Widgets
selected_tickers = st.sidebar.multiselect(
    "Select Tickers",
    options=available_tickers,
    key="ticker_select",
    help="Select ETFs to include in the analysis"
)

st.sidebar.text_input(
    "Add New Ticker",
    placeholder="e.g., SPY, QQQ",
    help="Enter ticker symbol and press Enter to add",
    key="new_ticker_input",
    on_change=add_new_ticker
)

st.sidebar.button("Reset to Default Tickers", on_click=reset_tickers)

thresh = st.sidebar.slider(
    "Min Valid Tickers",
    min_value=1, max_value=20, value=DEFAULT_THRESH,
    help="Minimum number of valid tickers required to run the analysis"
)

# Signal parameters
st.sidebar.subheader("Signal Window Parameters")
short_window = st.sidebar.slider(
    "Short Window (months)",
    min_value=1, max_value=12,
    key="short_window"
)
mid_window = st.sidebar.slider(
    "Mid Window (months)",
    min_value=1, max_value=12,
    key="mid_window"
)
long_window = st.sidebar.slider(
    "Long Window (months)",
    min_value=1, max_value=12,
    key="long_window"
)

# Weight parameters
st.sidebar.subheader("Signal Weight Parameters")
weight_mode = st.sidebar.radio(
    "Weight Calculation",
    ["Equal (1/3 each)", "Custom"],
    key="weight_mode",
)

if weight_mode == "Custom":
    short_wgt = st.sidebar.slider("Short Weight", 0.0, 1.0, step=0.01, key="short_wgt")
    mid_wgt = st.sidebar.slider("Mid Weight", 0.0, 1.0, step=0.01, key="mid_wgt")
    long_wgt = st.sidebar.slider("Long Weight", 0.0, 1.0, step=0.01, key="long_wgt")
    # Normalize weights
    total_wgt = short_wgt + mid_wgt + long_wgt
    if total_wgt > 0:
        short_wgt, mid_wgt, long_wgt = short_wgt/total_wgt, mid_wgt/total_wgt, long_wgt/total_wgt
else:
    short_wgt = mid_wgt = long_wgt = 1/3

# Portfolio parameters
st.sidebar.subheader("Portfolio Settings")
strategy_type = st.sidebar.selectbox(
    "Strategy Type",
    ["Top-K", "Quantile"],
    index=0
)

# Top-K options with select all
select_all_tickers = st.sidebar.checkbox(
    "Select All Tickers",
    value=False,
    help="Select all available tickers at each rebalancing (dynamic K)"
)

if select_all_tickers:
    top_k = None  # None means select all
    st.sidebar.info("All available tickers will be selected each period")
else:
    top_k = st.sidebar.slider(
        "Top K (for Top-K strategy)",
        min_value=1, max_value=10, value=DEFAULT_TOP_K
    )

n_quantiles = st.sidebar.slider(
    "Number of Quantiles",
    min_value=3, max_value=10, value=5
)

# Weight method selector
weight_method_label = st.sidebar.selectbox(
    "Weighting Method",
    options=list(WEIGHT_METHODS.keys()),
    index=0,
    help="Equal: 1/N weight | Inverse Vol: weight by 1/volatility | Rank: weight by 1/rank"
)
weight_method = WEIGHT_METHODS[weight_method_label]

tcost = st.sidebar.number_input(
    "Transaction Cost (one-way)",
    min_value=0.0, max_value=0.01, value=0.0, step=0.0005, format="%.4f"
)

# Benchmark settings
st.sidebar.subheader("Benchmark Settings")
bm_type = st.sidebar.radio(
    "Benchmark Type",
    ["Equal Weight", "Custom Ticker"],
    index=0,
    help="EW uses equal weight of all selected tickers"
)

# Initialize benchmark ticker in session state
if 'bm_ticker' not in st.session_state:
    st.session_state['bm_ticker'] = "SPY"

def apply_bm_ticker():
    new_val = st.session_state.get('bm_ticker_input', '').strip().upper()
    if new_val:
        st.session_state['bm_ticker'] = new_val

custom_bm_ticker = None
if bm_type == "Custom Ticker":
    st.sidebar.text_input(
        "Benchmark Ticker",
        value=st.session_state['bm_ticker'],
        placeholder="e.g., SPY, QQQ",
        help="Enter ticker symbol and press Enter to apply",
        key="bm_ticker_input",
        on_change=apply_bm_ticker
    )
    custom_bm_ticker = st.session_state['bm_ticker']
    st.sidebar.success(f"✓ Benchmark: **{custom_bm_ticker}**")

# Optimize button
st.sidebar.subheader("Parameter Optimization")
st.sidebar.caption("Optimize **Top-K strategy**. Uses Backtest Start Date, Top-K, Weighting Method, and Transaction Cost settings.")

opt_mode = st.sidebar.radio(
    "Search Mode",
    ["Full Grid", "Random Grid"],
    index=1,
    horizontal=True,
    help="Full Grid: 56 lookback combos (short<mid<long from 1-12mo) × 36 weight combos (10% step, sum=100%) = 2,016"
)

if opt_mode == "Random Grid":
    col_samples, col_seed = st.sidebar.columns(2)
    with col_samples:
        n_samples_input = st.text_input("Samples", value="500")
        try:
            n_samples = int(n_samples_input)
            n_samples = max(100, min(2016, n_samples))
        except ValueError:
            n_samples = 500
    with col_seed:
        seed_input = st.text_input("Seed", value="", help="Random seed for reproducibility (leave empty for random)", key="opt_seed")
        opt_seed = int(seed_input) if seed_input.strip().isdigit() else None
else:
    n_samples = None
    opt_seed = None

if st.sidebar.button("🔍 Optimize Parameters"):
    if len(selected_tickers) < 2:
        st.sidebar.error("Select at least 2 tickers first.")
    else:
        # Mark as running (will be checked on next page load if stopped)
        st.session_state['opt_running'] = True

        with st.spinner("Loading data for optimization..."):
            opt_dataset, _ = load_price_data(
                selected_tickers,
                start_date=str(start_date),
            )

        progress_bar = st.sidebar.progress(0, text="Optimizing...")

        def update_progress(current, total):
            progress = current / total
            progress_bar.progress(progress, text=f"Testing {current}/{total} combinations...")

        with st.spinner("Running optimization..."):
            result = optimize_sharpe(
                price=opt_dataset,
                top_k=None if select_all_tickers else top_k,
                weight_method=weight_method,
                tcost=tcost,
                start_date=str(backtest_start_date),
                thresh=thresh,
                n_samples=n_samples,
                seed=opt_seed,
                progress_callback=update_progress,
            )

        # Only reaches here if completed (not stopped)
        st.session_state['opt_running'] = False
        progress_bar.empty()

        if result['best_params'] is not None:
            bp = result['best_params']
            # Store optimized params for next render cycle
            st.session_state['optimized_params'] = bp
            st.session_state['optimization_result'] = {
                'sharpe': result['best_sharpe'],
                'tested': result['total_combinations'],
                'total': result['total_possible'],
            }
            st.rerun()
        else:
            st.sidebar.error("Optimization failed. Try different settings.")

# Show optimization result message
if 'optimization_result' in st.session_state:
    res = st.session_state['optimization_result']
    tested_info = f"{res['tested']}/{res['total']}"
    st.sidebar.success(f"""
    **Optimization Complete!** (tested {tested_info})
    - Sharpe: {res['sharpe']:.3f}
    - Windows: {st.session_state['short_window']}/{st.session_state['mid_window']}/{st.session_state['long_window']} mo
    - Weights: {st.session_state['short_wgt']:.0%}/{st.session_state['mid_wgt']:.0%}/{st.session_state['long_wgt']:.0%}

    Parameters updated. Click **Run Analysis** to apply.
    """)
    del st.session_state['optimization_result']

# Walk-Forward Optimization
st.sidebar.subheader("Walk-Forward Analysis")
st.sidebar.caption("Out-of-sample validation. Uses Download Start Date.")

with st.sidebar.expander("Walk-Forward Settings"):
    wf_window_type = st.radio(
        "Window Type",
        ["Rolling", "Expanding"],
        index=0,
        horizontal=True,
        help="Rolling: Fixed train window. Expanding: Train from start, grows each fold."
    )
    if wf_window_type == "Rolling":
        wf_train = st.number_input("Train Period (months)", min_value=12, value=36, step=12)
    else:
        wf_train = st.number_input("Min Train Period (months)", min_value=12, value=36, step=12,
                                   help="Minimum training period for the first fold")
    wf_test = st.number_input("Test Period (months)", min_value=1, value=12, step=1)
    wf_step = st.number_input("Step Size (months)", min_value=1, value=12, step=1)
    wf_samples = st.number_input("Samples per Fold", min_value=100, max_value=5000, value=500, step=100)
    wf_seed_input = st.text_input("Seed", value="", help="Random seed for reproducibility (leave empty for random)", key="wf_seed")
    wf_seed = int(wf_seed_input) if wf_seed_input.strip().isdigit() else None

if st.sidebar.button("🔄 Walk-Forward Optimize"):
    if len(selected_tickers) < 2:
        st.sidebar.error("Select at least 2 tickers first.")
    else:
        # Mark as running (will be checked on next page load if stopped)
        st.session_state['wf_running'] = True

        # Clear previous walk-forward result to avoid stale data on stop
        if 'walk_forward_result' in st.session_state:
            del st.session_state['walk_forward_result']

        # Get benchmark settings from session state
        wf_bm_ticker = st.session_state.get('bm_ticker', None)
        wf_bm_data = None

        with st.spinner("Loading data for walk-forward..."):
            wf_dataset, _ = load_price_data(
                selected_tickers,
                start_date=str(start_date),
            )

            # Load benchmark data if custom ticker is set
            if wf_bm_ticker and wf_bm_ticker not in wf_dataset.columns:
                wf_bm_dataset, _ = load_price_data(
                    [wf_bm_ticker],
                    start_date=str(start_date),
                )
                if wf_bm_ticker in wf_bm_dataset.columns:
                    wf_bm_data = wf_bm_dataset[wf_bm_ticker]

        wf_progress_bar = st.sidebar.progress(0, text="Walk-Forward Optimization...")
        wf_status = st.sidebar.empty()

        def wf_update_progress(fold, total_folds, step, total_steps):
            # total_folds + 1 means final training
            if fold > total_folds:
                overall = (total_folds * total_steps + step) / ((total_folds + 1) * total_steps)
                wf_progress_bar.progress(min(overall, 0.99), text=f"Final Training - {step}/{total_steps}")
                wf_status.caption("Training on latest data for recommended params...")
            else:
                overall = ((fold - 1) * total_steps + step) / ((total_folds + 1) * total_steps)
                wf_progress_bar.progress(overall, text=f"Fold {fold}/{total_folds} - {step}/{total_steps}")
                wf_status.caption(f"Training fold {fold} of {total_folds}...")

        with st.spinner("Running walk-forward optimization..."):
            wf_result = walk_forward_optimize(
                price=wf_dataset,
                train_months=wf_train,
                test_months=wf_test,
                step_months=wf_step,
                window_type=wf_window_type.lower(),
                top_k=None if select_all_tickers else top_k,
                weight_method=weight_method,
                tcost=tcost,
                thresh=thresh,
                n_samples=wf_samples,
                seed=wf_seed,
                bm_ticker=wf_bm_ticker,
                bm_data=wf_bm_data,
                progress_callback=wf_update_progress,
            )

        # Only reaches here if completed (not stopped)
        st.session_state['wf_running'] = False
        wf_progress_bar.empty()
        wf_status.empty()

        if 'error' in wf_result:
            st.sidebar.error(wf_result['error'])
        else:
            # Save benchmark settings used during walk-forward execution
            wf_result['_bm_type'] = bm_type
            wf_result['_bm_ticker'] = wf_bm_ticker if bm_type == "Custom Ticker" else None
            wf_result['_bm_data'] = wf_bm_data
            st.session_state['walk_forward_result'] = wf_result
            st.rerun()

# Show walk-forward result
if 'walk_forward_result' in st.session_state:
    wf = st.session_state['walk_forward_result']
    decay = wf['sharpe_decay']
    decay_str = f"{decay:+.2f}" if not np.isnan(decay) else "N/A"
    wf_type = wf.get('window_type', 'rolling').capitalize()

    st.sidebar.info(f"""
    **Walk-Forward Complete!** ({wf['valid_folds']}/{wf['total_folds']} folds, {wf_type})
    - In-Sample Sharpe (avg): **{wf['is_sharpe_avg']:.2f}**
    - Out-of-Sample Sharpe (avg): **{wf['oos_sharpe_avg']:.2f}**
    - Combined OOS Sharpe: **{wf['oos_sharpe']:.2f}**
    - Sharpe Decay: **{decay_str}**

    {'⚠️ High decay suggests overfitting!' if decay > 0.5 else '✓ Reasonable stability' if not np.isnan(decay) else ''}
    """)

    # Show final recommended params
    fp = wf.get('final_params')
    if fp:
        ftp = wf.get('final_train_period', {})
        st.sidebar.success(f"""
        **Recommended Params** (trained on {ftp.get('start', '?')[:7]} ~ {ftp.get('end', '?')[:7]})
        - Windows: **{fp['short_window']}/{fp['mid_window']}/{fp['long_window']}** mo
        - Weights: **{fp['short_wgt']:.0%}/{fp['mid_wgt']:.0%}/{fp['long_wgt']:.0%}**
        - Sharpe: **{wf.get('final_sharpe', 0):.2f}**
        """)

        if st.sidebar.button("✅ Apply Recommended Params"):
            st.session_state['optimized_params'] = fp
            st.rerun()

# =============================================================================
# Main Content
# =============================================================================

# Load data button
if st.sidebar.button("🔄 Run Analysis", type="primary"):
    if len(selected_tickers) < 2:
        st.error("Please select at least 2 tickers for analysis.")
    else:
        with st.spinner("Loading price data..."):
            try:
                # Load main dataset
                dataset, missing_tickers = load_price_data(
                    selected_tickers,
                    start_date=str(start_date),
                )

                # Load custom benchmark ticker if specified
                bm_data = None
                bm_missing = False
                if custom_bm_ticker and custom_bm_ticker not in dataset.columns:
                    bm_dataset, bm_missing_list = load_price_data(
                        [custom_bm_ticker],
                        start_date=str(start_date),
                    )
                    if custom_bm_ticker in bm_dataset.columns:
                        bm_data = bm_dataset[custom_bm_ticker]
                    else:
                        bm_missing = True

                st.session_state['dataset'] = dataset
                st.session_state['bm_data'] = bm_data
                st.session_state['params'] = {
                    'thresh': thresh,
                    'short_window': short_window,
                    'mid_window': mid_window,
                    'long_window': long_window,
                    'short_wgt': short_wgt,
                    'mid_wgt': mid_wgt,
                    'long_wgt': long_wgt,
                    'top_k': top_k,
                    'n_quantiles': n_quantiles,
                    'weight_method': weight_method,
                    'tcost': tcost,
                    'strategy_type': strategy_type,
                    'backtest_start_date': str(backtest_start_date),
                    'custom_bm_ticker': custom_bm_ticker if bm_type == "Custom Ticker" else None,
                }
                st.success(f"Loaded {len(dataset)} days of data for {len(dataset.columns)} tickers")
                if missing_tickers:
                    st.warning(f"Data not found for: {', '.join(missing_tickers)}")
                if bm_missing:
                    st.warning(f"Benchmark ticker '{custom_bm_ticker}' not found. Using Equal Weight BM instead.")
            except Exception as e:
                st.error(f"Error loading data: {e}")

# Display analysis if data is loaded
if 'dataset' in st.session_state:
    dataset = st.session_state['dataset']
    params = st.session_state['params']
    bm_data = st.session_state.get('bm_data')

    # Generate signal
    with st.spinner("Generating signals..."):
        signal = generate_signal(
            dataset,
            short_window=params['short_window'],
            mid_window=params['mid_window'],
            long_window=params['long_window'],
            short_wgt=params['short_wgt'],
            mid_wgt=params['mid_wgt'],
            long_wgt=params['long_wgt'],
            thresh=params['thresh'],
        )

    # Run backtest based on strategy type
    with st.spinner("Running backtest..."):
        if params['strategy_type'] == "Top-K":
            strat_nav, universe_bm_nav, turnover = run_top_k_backtest(
                dataset, signal,
                top_k=params['top_k'],
                weight_method=params['weight_method'],
                tcost=params['tcost'],
            )
            # Strategy name
            strat_name = "All" if params['top_k'] is None else f"Top-{params['top_k']}"
            navs = pd.DataFrame({strat_name: strat_nav})
        else:
            q_nav, q_to, universe_bm_nav = run_quantile_backtest(
                dataset, signal,
                n_quantiles=params['n_quantiles'],
                tcost=params['tcost'],
                weight_method=params['weight_method'],
            )
            navs = q_nav.copy()

        # Handle benchmark
        if params.get('custom_bm_ticker') and bm_data is not None:
            # Use custom ticker as benchmark
            bm_ticker = params['custom_bm_ticker']
            # Align benchmark data with strategy NAV dates
            bm_aligned = bm_data.reindex(navs.index).ffill().bfill()
            # Convert price to NAV (starting at same value as strategy)
            bm_nav = bm_aligned / bm_aligned.iloc[0] * 1000
            bm_nav.name = bm_ticker
            navs['BM'] = bm_nav
        else:
            # Use universe equal weight as benchmark
            navs['BM'] = universe_bm_nav

    # Apply backtest start date for display
    backtest_start = pd.to_datetime(params['backtest_start_date'])
    # Find actual start date (use available data if backtest_start is too early)
    actual_start = navs.index[navs.index >= backtest_start]
    if len(actual_start) > 0:
        display_start = actual_start[0]
    else:
        display_start = navs.index[0]
        st.warning(f"Backtest start date {backtest_start.strftime('%Y-%m-%d')} is after available data. Using {display_start.strftime('%Y-%m-%d')} instead.")

    # Filter and rebase NAVs for display
    navs_display = navs[display_start:].copy()
    navs_rebased = rebase(navs_display)

    # ==========================================================================
    # User Guide (collapsible)
    # ==========================================================================
    with st.expander("📖 User Guide", expanded=False):
        st.markdown("""
        ### Quick Reference

        #### Signal Parameters
        - **Signal Window**: Lookback period in months for each timeframe
        - **Signal Weight**: Weight for each window (Equal 1/3 or User-defined weights)

        #### Strategy Types
        - **Top-K**: Invests in top K ranked assets
        - **Quantile**: Divides assets into groups to analyze momentum factor

        #### Weighting Methods
        - **Equal**: 1/N weight for each asset
        - **Inverse Vol**: Higher weight to lower volatility assets
        - **Rank**: Higher weight to better ranked assets

        #### Tips
        - Use **Optimize Parameters** to find the best lookback windows and weights
        """)

    # ==========================================================================
    # KPI Section
    # ==========================================================================
    st.subheader("Key Performance Indicators")

    # Show backtest period info
    st.caption(f"Backtest Period: {display_start.strftime('%Y-%m-%d')} ~ {navs_display.index[-1].strftime('%Y-%m-%d')}")

    kpis = calculate_kpis(navs_display)
    cols = st.columns(len(kpis))

    for i, (name, metrics) in enumerate(kpis.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            st.metric("CAGR", f"{metrics['cagr']:.1%}")
            st.metric("Volatility", f"{metrics['vol']:.1%}")
            st.metric("Sharpe", f"{metrics['sharpe']:.2f}")
            st.metric("MDD", f"{metrics['mdd']:.1%}")
            st.metric("YTD", f"{metrics['ytd']:.1%}")

    # ==========================================================================
    # Charts Section
    # ==========================================================================
    st.header("Performance Charts")

    # Chart config: only keep download and fullscreen buttons
    chart_config = {
        'displayModeBar': True,
        'modeBarButtonsToRemove': [
            'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
            'autoScale2d', 'resetScale2d', 'hoverClosestCartesian',
            'hoverCompareCartesian', 'toggleSpikelines'
        ],
        'modeBarButtonsToAdd': [],
        'displaylogo': False,
    }

    # NAV Chart (already rebased above)
    fig_nav = create_nav_chart(navs_rebased, title="Portfolio NAV (Rebased to 100)")
    st.plotly_chart(fig_nav, use_container_width=True, config=chart_config)

    # Drawdown Chart (full width to match NAV chart)
    fig_dd = create_drawdown_chart(navs_rebased, title="Drawdown (%)", height=350)
    st.plotly_chart(fig_dd, use_container_width=True, config=chart_config)

    # Signal Category Table (Q5~Q1 with all tickers)
    latest_signal = get_signal_ranking(signal)
    st.subheader(f"Current Signal by Quantile ({dataset.index[-1].strftime('%Y-%m-%d')})")
    st.caption("Rank 1 = Best (Q5), Higher Rank = Worse (Q1). Format: Ticker (Rank)")
    fig_category = create_signal_category_table(
        latest_signal,
        n_quantiles=params['n_quantiles'],
        title=""
    )
    st.plotly_chart(fig_category, use_container_width=True, config=chart_config)

    # Raw Signal Data
    with st.expander("Show Raw Signal Data"):
        st.subheader("Latest Signal Ranking")
        signal_display = signal.iloc[-5:].T.copy()
        signal_display.columns = signal_display.columns.strftime('%Y-%m-%d')
        st.dataframe(signal_display.style.format("{:.0f}"))

    # Quantile Spread (only for Quantile strategy)
    if params['strategy_type'] == "Quantile" and 'Q1' in navs_display.columns and 'Q5' in navs_display.columns:
        st.subheader("Quantile Spread (Q5 - Q1)")
        fig_spread = create_quantile_spread_chart(navs_display)
        st.plotly_chart(fig_spread, use_container_width=True, config=chart_config)

    # ==========================================================================
    # Holdings Heatmap
    # ==========================================================================
    st.header("Holdings Analysis")

    # Get weights for heatmap
    if params['strategy_type'] == "Top-K":
        wgt, _ = compute_weight_top_k(
            dataset, signal,
            top_k=params['top_k'],
            weight_method=params['weight_method']
        )
        selected_quantile = "All" if params['top_k'] is None else f"Top-{params['top_k']}"
    else:
        from src.portfolio import compute_weight_quantile
        q_wgts, _ = compute_weight_quantile(
            dataset, signal,
            n_quantiles=params['n_quantiles'],
            weight_method=params['weight_method']
        )

        # Quantile selector (default Q5 = best performers)
        quantile_options = [f"Q{q}" for q in range(params['n_quantiles'], 0, -1)]
        col_select, col_info = st.columns([1, 3])
        with col_select:
            selected_quantile = st.selectbox(
                "Select Quantile",
                options=quantile_options,
                index=0,  # Q5 (best) as default
                help="Q5 = Best performers (Rank 1~), Q1 = Worst performers"
            )
        with col_info:
            n_q = params['n_quantiles']
            if selected_quantile == f'Q{n_q}':
                desc = "Best performers (lowest ranks)"
            elif selected_quantile == 'Q1':
                desc = "Worst performers (highest ranks)"
            else:
                desc = "Middle performers"
            st.info(f"📈 **{selected_quantile}**: {desc}")

        wgt = q_wgts[selected_quantile]

    # Limit to recent data (36 months = 3 years)
    wgt.index = wgt.index + BMonthBegin(1)
    recent_wgt = wgt.iloc[-36:]
    n_tickers = len(recent_wgt.columns)
    chart_height = max(500, n_tickers * 25)  # Ensure all tickers visible
    fig_heatmap = create_holding_heatmap(
        recent_wgt,
        title=f"Monthly Holdings - {selected_quantile} (Last 36 Months)",
        height=chart_height,
    )
    st.plotly_chart(fig_heatmap, use_container_width=True, config=chart_config)

    # ==========================================================================
    # Statistics Table
    # ==========================================================================
    st.header("Performance Statistics")

    stats = summary_stats(navs_display)
    fig_table = create_returns_table(stats)
    st.plotly_chart(fig_table, use_container_width=True, config=chart_config)

    # Also show as DataFrame for easy copying
    with st.expander("Show as DataFrame"):
        st.dataframe(stats.T.style.format({
            'nyears': '{:.1f}',
            'nsamples': '{:.0f}',
            'cumulative': '{:.2%}',
            'cagr': '{:.2%}',
            'mean': '{:.2%}',
            'vol': '{:.2%}',
            'sharpe': '{:.2f}',
            'mdd': '{:.2%}',
            'max': '{:.2%}',
            'min': '{:.2%}',
            'skew': '{:.2f}',
            'kurt': '{:.2f}',
        }))

    # ==========================================================================
    # Walk-Forward Analysis Results
    # ==========================================================================
    if 'walk_forward_result' in st.session_state:
        wf = st.session_state['walk_forward_result']
        if 'error' not in wf and wf['valid_folds'] > 0:
            wf_type = wf.get('window_type', 'rolling').capitalize()
            st.header(f"Walk-Forward Analysis ({wf_type} Window)")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("In-Sample Sharpe (avg)", f"{wf['is_sharpe_avg']:.2f}")
            with col2:
                st.metric("Out-of-Sample Sharpe (avg)", f"{wf['oos_sharpe_avg']:.2f}")
            with col3:
                st.metric("Combined OOS Sharpe", f"{wf['oos_sharpe']:.2f}")
            with col4:
                decay = wf['sharpe_decay']
                decay_str = f"{decay:.2f}" if not np.isnan(decay) else "N/A"
                st.metric("Sharpe Decay", decay_str,
                         delta=None if np.isnan(decay) else f"{-decay:.2f}",
                         delta_color="inverse")

            # Interpretation
            if not np.isnan(wf['sharpe_decay']):
                if wf['sharpe_decay'] > 0.5:
                    st.warning("⚠️ **High Sharpe Decay**: In-sample performance significantly exceeds out-of-sample. This suggests potential overfitting.")
                elif wf['sharpe_decay'] > 0.2:
                    st.info("ℹ️ **Moderate Sharpe Decay**: Some performance degradation out-of-sample. Consider simpler parameter choices.")
                else:
                    st.success("✓ **Low Sharpe Decay**: Strategy shows reasonable stability between in-sample and out-of-sample performance.")

            # Fold-by-fold results table
            st.subheader("Fold Results")
            fold_df = pd.DataFrame([
                {
                    'Fold': f['fold'],
                    'Train Period': f"{f['train_start'][:7]} ~ {f['train_end'][:7]}",
                    'Test Period': f"{f['test_start'][:7]} ~ {f['test_end'][:7]}",
                    'Windows': f"{f['params']['short_window']}/{f['params']['mid_window']}/{f['params']['long_window']}",
                    'Weights': f"{f['params']['short_wgt']:.0%}/{f['params']['mid_wgt']:.0%}/{f['params']['long_wgt']:.0%}",
                    'IS Sharpe': f['is_sharpe'],
                    'OOS Sharpe': f['oos_sharpe'],
                }
                for f in wf['folds']
            ])
            st.dataframe(
                fold_df.style.format({
                    'IS Sharpe': '{:.2f}',
                    'OOS Sharpe': '{:.2f}',
                }),
                use_container_width=True,
                hide_index=True
            )

            # Final recommended params (trained on latest data)
            fp = wf.get('final_params')
            if fp:
                st.subheader("Recommended Parameters")
                ftp = wf.get('final_train_period', {})
                st.caption(f"Trained on latest data: {ftp.get('start', '?')[:7]} ~ {ftp.get('end', '?')[:7]}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Windows (S/M/L)",
                             f"{fp['short_window']}/{fp['mid_window']}/{fp['long_window']} mo")
                with col2:
                    st.metric("Weights (S/M/L)",
                             f"{fp['short_wgt']:.0%}/{fp['mid_wgt']:.0%}/{fp['long_wgt']:.0%}")
                with col3:
                    st.metric("In-Sample Sharpe", f"{wf.get('final_sharpe', 0):.2f}")

                st.info("💡 Use **Apply Recommended Params** button in sidebar to apply these parameters.")

            # Parameter stability analysis
            if wf['param_stability']:
                st.subheader("Parameter Stability")
                st.caption("How much do optimal parameters vary across folds? Lower std = more stable.")

                stability_df = pd.DataFrame(wf['param_stability']).T
                stability_df.index = ['Short Window', 'Mid Window', 'Long Window', 'Short Weight', 'Mid Weight', 'Long Weight']
                stability_df.columns = ['Mean', 'Std', 'Min', 'Max']

                # Format weights as percentages
                for idx in ['Short Weight', 'Mid Weight', 'Long Weight']:
                    stability_df.loc[idx] = stability_df.loc[idx] * 100

                st.dataframe(
                    stability_df.style.format({
                        'Mean': '{:.1f}',
                        'Std': '{:.1f}',
                        'Min': '{:.1f}',
                        'Max': '{:.1f}',
                    }),
                    use_container_width=True
                )

            # Combined OOS NAV chart
            if wf['combined_oos_nav'] is not None:
                st.subheader("Combined Out-of-Sample NAV")

                # Use benchmark settings saved during walk-forward execution
                wf_saved_bm_type = wf.get('_bm_type', 'Equal Weight')
                wf_saved_bm_ticker = wf.get('_bm_ticker')
                wf_saved_bm_data = wf.get('_bm_data')
                wf_bm_name = wf_saved_bm_ticker if wf_saved_bm_type == "Custom Ticker" and wf_saved_bm_ticker else "Equal Weight"
                st.caption(f"NAV from chaining all out-of-sample test periods. BM = {wf_bm_name}")

                # Rebase to 100 for consistency with main chart
                oos_nav_rebased = wf['combined_oos_nav'] / wf['combined_oos_nav'].iloc[0] * 100

                import plotly.graph_objects as go
                fig_oos = go.Figure()
                fig_oos.add_trace(go.Scatter(
                    x=oos_nav_rebased.index,
                    y=oos_nav_rebased.values,
                    mode='lines',
                    name='Strategy (OOS)',
                    line=dict(color='#2196F3', width=2)
                ))

                # Compute benchmark NAV using settings saved during walk-forward execution
                oos_bm_nav = None
                oos_date_range = wf.get('oos_date_range')
                if oos_date_range:
                    oos_start = oos_date_range['start']
                    oos_end = oos_date_range['end']

                    if wf_saved_bm_type == "Custom Ticker" and wf_saved_bm_data is not None:
                        # Custom ticker: buy and hold
                        bm_slice = wf_saved_bm_data[oos_start:oos_end]
                        if len(bm_slice) > 0:
                            oos_bm_nav = bm_slice / bm_slice.iloc[0] * 100
                    else:
                        # Equal weight: compute from dataset
                        dataset_slice = dataset[oos_start:oos_end]
                        if len(dataset_slice) > 0:
                            # Simple equal weight daily return
                            daily_ret = dataset_slice.pct_change().fillna(0)
                            ew_ret = daily_ret.mean(axis=1)
                            oos_bm_nav = (1 + ew_ret).cumprod() * 100

                # Add benchmark if computed
                if oos_bm_nav is not None and len(oos_bm_nav) > 0:
                    fig_oos.add_trace(go.Scatter(
                        x=oos_bm_nav.index,
                        y=oos_bm_nav.values,
                        mode='lines',
                        name=f'BM ({wf_bm_name})',
                        line=dict(color='#9E9E9E', width=1.5, dash='dash')
                    ))

                fig_oos.update_layout(
                    title="Walk-Forward Out-of-Sample Performance",
                    xaxis_title="Date",
                    yaxis_title="NAV",
                    template="plotly_white",
                    height=400,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig_oos, use_container_width=True, config=chart_config)

                # Show OOS Sharpe comparison
                oos_sharpe = wf['oos_sharpe']
                if oos_bm_nav is not None and len(oos_bm_nav) > 1:
                    bm_daily_ret = oos_bm_nav.pct_change().dropna()
                    if len(bm_daily_ret) > 0 and bm_daily_ret.std() > 0:
                        oos_bm_sharpe = bm_daily_ret.mean() / bm_daily_ret.std() * np.sqrt(252)
                        excess = oos_sharpe - oos_bm_sharpe
                        st.caption(f"OOS Sharpe: Strategy **{oos_sharpe:.2f}** vs BM **{oos_bm_sharpe:.2f}** (Excess: {excess:+.2f})")

else:
    # Initial state - show instructions
    st.info("👈 Configure parameters in the sidebar and click **Run Analysis** to start.")

    st.markdown("""
    ## User Guide

    This dashboard analyzes **momentum-based asset rotation strategies** across ETFs.

    ---

    ### Quick Start
    1. Select tickers in the sidebar (or use defaults)
    2. Click **Run Analysis**
    3. Review the results

    ---

    ### Signal Parameters
    Momentum is measured using 3 lookback periods (Short/Mid/Long).  
    For each period, assets are ranked by returns and the final signal is a weighted average of these ranks.
    - **Signal Window**: Lookback period in months for each timeframe
    - **Signal Weight**: Weight for each window (Equal 1/3 or User-defined weights)
                
    ---
                               
    ### Portfolio Settings
    **Strategy Type**
    - **Top-K**: Invests in top K ranked assets
    - **Quantile**: Divides assets into groups to analyze momentum factor

    **Weighting Method**
    - **Equal**: 1/N weight for each asset
    - **Inverse Vol**: Higher weight to lower volatility assets
    - **Rank**: Higher weight to better ranked assets

    **Transaction Cost**: One-way cost (0.001 = 0.1%)

    ---                

    ### Parameter Optimization
    Find optimal parameter combinations:
    - **Full Grid**: Tests all combinations (2,016 total)
    - **Random Grid**: Random sampling (default 500, faster)

    ---
                                
    ### Benchmark Settings
    - **Equal Weight**: Hold all selected tickers with equal weight
    - **Custom Ticker**: Use a specific ticker (e.g., SPY) as benchmark

    """)
