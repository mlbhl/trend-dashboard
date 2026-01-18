"""Trend Dashboard - Momentum Strategy Analysis App"""

import streamlit as st
import pandas as pd
from pandas.tseries.offsets import BMonthBegin

from src.config import (
    ALPHA_LIST,
    TICKER_DESCRIPTIONS,
    DEFAULT_START_DATE,
    DEFAULT_SHORT_WINDOW,
    DEFAULT_MID_WINDOW,
    DEFAULT_LONG_WINDOW,
    DEFAULT_SHORT_WEIGHT,
    DEFAULT_MID_WEIGHT,
    DEFAULT_LONG_WEIGHT,
    DEFAULT_TOP_K,
    DEFAULT_TCOST,
    DEFAULT_THRESH,
)
from src.data import load_price_data
from src.signals import generate_signal, get_signal_ranking
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
    min_value=pd.to_datetime("2000-01-01"),
)

# Proxy setting (optional)
use_proxy = st.sidebar.checkbox("Use Proxy", value=False)
proxy_url = None
if use_proxy:
    proxy_url = st.sidebar.text_input(
        "Proxy URL",
        placeholder="http://proxy.example.com:8080"
    )

# Ticker selection
st.sidebar.subheader("Ticker Selection")

# Initialize custom tickers in session state
if 'custom_tickers' not in st.session_state:
    st.session_state['custom_tickers'] = []

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
st.sidebar.subheader("Signal Parameters")
short_window = st.sidebar.slider(
    "Short Window (months)",
    min_value=1, max_value=12, value=DEFAULT_SHORT_WINDOW
)
mid_window = st.sidebar.slider(
    "Mid Window (months)",
    min_value=1, max_value=12, value=DEFAULT_MID_WINDOW
)
long_window = st.sidebar.slider(
    "Long Window (months)",
    min_value=1, max_value=12, value=DEFAULT_LONG_WINDOW
)

# Weight parameters
st.sidebar.subheader("Weight Parameters")
weight_mode = st.sidebar.radio(
    "Weight Calculation",
    ["Equal (1/3 each)", "Custom"],
    index=1
)

if weight_mode == "Custom":
    short_wgt = st.sidebar.slider("Short Weight", 0.0, 1.0, DEFAULT_SHORT_WEIGHT, 0.01)
    mid_wgt = st.sidebar.slider("Mid Weight", 0.0, 1.0, DEFAULT_MID_WEIGHT, 0.01)
    long_wgt = st.sidebar.slider("Long Weight", 0.0, 1.0, DEFAULT_LONG_WEIGHT, 0.01)
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

top_k = st.sidebar.slider(
    "Top K (for Top-K strategy)",
    min_value=1, max_value=10, value=DEFAULT_TOP_K
)

n_quantiles = st.sidebar.slider(
    "Number of Quantiles",
    min_value=3, max_value=10, value=5
)

inverse_vol = st.sidebar.checkbox("Inverse Volatility Weighting", value=False)

tcost = st.sidebar.number_input(
    "Transaction Cost (one-way)",
    min_value=0.0, max_value=0.01, value=0.0, step=0.0005, format="%.4f"
)

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
                dataset, missing_tickers = load_price_data(
                    selected_tickers,
                    start_date=str(start_date),
                    proxy=proxy_url if use_proxy and proxy_url else None,
                )
                st.session_state['dataset'] = dataset
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
                    'inverse_vol': inverse_vol,
                    'tcost': tcost,
                    'strategy_type': strategy_type,
                }
                st.success(f"Loaded {len(dataset)} days of data for {len(dataset.columns)} tickers")
                if missing_tickers:
                    st.warning(f"Data not found for: {', '.join(missing_tickers)}")
            except Exception as e:
                st.error(f"Error loading data: {e}")

# Display analysis if data is loaded
if 'dataset' in st.session_state:
    dataset = st.session_state['dataset']
    params = st.session_state['params']

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
            strat_nav, bm_nav, turnover = run_top_k_backtest(
                dataset, signal,
                top_k=params['top_k'],
                inverse_vol=params['inverse_vol'],
                tcost=params['tcost'],
            )
            navs = pd.DataFrame({
                f"Top-{params['top_k']}": strat_nav,
                'BM': bm_nav
            })
        else:
            q_nav, q_to, bm_nav = run_quantile_backtest(
                dataset, signal,
                n_quantiles=params['n_quantiles'],
                tcost=params['tcost'],
            )
            navs = pd.concat([q_nav, bm_nav], axis=1)

    # ==========================================================================
    # KPI Section
    # ==========================================================================
    st.subheader("Key Performance Indicators")

    kpis = calculate_kpis(navs)
    cols = st.columns(len(kpis))

    for i, (name, metrics) in enumerate(kpis.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            st.metric("CAGR", f"{metrics['cagr']:.2%}")
            st.metric("Sharpe", f"{metrics['sharpe']:.2f}")
            st.metric("MDD", f"{metrics['mdd']:.2%}")
            st.metric("YTD", f"{metrics['ytd']:.2%}")

    # ==========================================================================
    # Charts Section
    # ==========================================================================
    st.header("Performance Charts")

    # NAV Chart
    navs_rebased = rebase(navs)
    fig_nav = create_nav_chart(navs_rebased, title="Portfolio NAV (Rebased to 100)")
    st.plotly_chart(fig_nav, use_container_width=True)

    # Drawdown Chart (full width to match NAV chart)
    fig_dd = create_drawdown_chart(navs_rebased, title="Drawdown (%)", height=350)
    st.plotly_chart(fig_dd, use_container_width=True)

    # Signal Category Table (Q5~Q1 with all tickers)
    latest_signal = get_signal_ranking(signal)
    st.subheader(f"Current Signal by Quantile ({signal.index[-1].strftime('%Y-%m-%d')})")
    st.caption("Rank 1 = Best (Q5), Higher Rank = Worse (Q1). Format: Ticker (Rank)")
    fig_category = create_signal_category_table(
        latest_signal,
        n_quantiles=params['n_quantiles'],
        title=""
    )
    st.plotly_chart(fig_category, use_container_width=True)

    # Quantile Spread (only for Quantile strategy)
    if params['strategy_type'] == "Quantile" and 'Q1' in navs.columns and 'Q5' in navs.columns:
        st.subheader("Quantile Spread (Q5 - Q1)")
        fig_spread = create_quantile_spread_chart(navs)
        st.plotly_chart(fig_spread, use_container_width=True)

    # ==========================================================================
    # Holdings Heatmap
    # ==========================================================================
    st.header("Holdings Analysis")

    # Get weights for heatmap
    if params['strategy_type'] == "Top-K":
        wgt, _ = compute_weight_top_k(
            dataset, signal,
            top_k=params['top_k'],
            inverse_vol=params['inverse_vol']
        )
        selected_quantile = f"Top-{params['top_k']}"
    else:
        from src.portfolio import compute_weight_quantile
        q_wgts, _ = compute_weight_quantile(
            dataset, signal,
            n_quantiles=params['n_quantiles'],
            inverse_vol=params['inverse_vol']
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
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ==========================================================================
    # Statistics Table
    # ==========================================================================
    st.header("Performance Statistics")

    stats = summary_stats(navs)
    fig_table = create_returns_table(stats)
    st.plotly_chart(fig_table, use_container_width=True)

    # Also show as DataFrame for easy copying
    with st.expander("Show as DataFrame"):
        st.dataframe(stats.T.style.format({
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
    # Raw Data Section
    # ==========================================================================
    with st.expander("Show Raw Signal Data"):
        st.subheader("Latest Signal Ranking")
        st.dataframe(signal.iloc[-5:].T.style.format("{:.0f}"))

else:
    # Initial state - show instructions
    st.info("👈 Configure parameters in the sidebar and click **Run Analysis** to start.")

    st.markdown("""
    ### How to Use
    1. **Select Tickers**: Choose the ETFs you want to analyze
    2. **Set Signal Parameters**: Adjust momentum windows (short/mid/long)
    3. **Choose Strategy**: Top-K selects best-ranked assets, Quantile divides into groups
    4. **Run Analysis**: Click the button to load data and run backtest

    ### About This Dashboard
    This dashboard implements a **momentum-based rotation strategy** across various ETFs.
    The signal is generated by ranking assets based on their momentum over multiple time horizons.
    """)
