"""Callback functions for Trend Dashboard Dash app."""

import json
import numpy as np
import pandas as pd
import dash
from dash import Input, Output, State, callback, html, no_update, dash_table, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from pandas.tseries.offsets import BMonthBegin

from src.config import TICKER_PRESETS, DEFAULT_PRESET, WEIGHT_METHODS
from src.data import load_price_data
from src.signals import generate_signal, get_signal_ranking
from src.portfolio import (
    compute_weight_top_k,
    compute_weight_quantile,
    run_top_k_backtest,
    run_quantile_backtest,
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
from src.optimizer import optimize_sharpe, walk_forward_optimize
from dash_components.layout import _create_bm_row


def _build_composite_bm(bm_config: list, start_date: str) -> tuple:
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
            label_parts.append(f"{t} {w:.0f}%")
    bm_label = " + ".join(label_parts)

    return composite_price, bm_label, warn_msgs


def register_callbacks(app):
    """Register all callbacks for the Dash app."""

    # =========================================================================
    # UI Toggle Callbacks
    # =========================================================================

    @app.callback(
        Output("sidebar-collapse", "is_open"),
        Input("mobile-sidebar-toggle", "n_clicks"),
        State("sidebar-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_mobile_sidebar(n_clicks, is_open):
        """Toggle mobile sidebar collapse."""
        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output("custom-weight-div", "style"),
        Input("weight-mode-radio", "value"),
    )
    def toggle_custom_weights(weight_mode):
        """Show/hide custom weight sliders based on weight mode."""
        if weight_mode == "custom":
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        Output("custom-bm-div", "style"),
        Input("bm-type-radio", "value"),
    )
    def toggle_custom_benchmark(bm_type):
        """Show/hide custom benchmark input based on benchmark type."""
        if bm_type == "custom":
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        Output("bm-rows-container", "children"),
        Input("bm-add-row-btn", "n_clicks"),
        Input({"type": "bm-row-delete", "index": ALL}, "n_clicks"),
        State("bm-rows-container", "children"),
        prevent_initial_call=True,
    )
    def manage_bm_rows(add_clicks, delete_clicks, current_rows):
        """Add or delete benchmark ticker rows."""
        from dash import ctx

        triggered = ctx.triggered_id

        if not current_rows:
            current_rows = []

        # Add row
        if triggered == "bm-add-row-btn":
            # Find next index (max existing + 1)
            existing_indices = []
            for row in current_rows:
                if isinstance(row, dict) and "props" in row:
                    row_id = row["props"].get("id", {})
                    if isinstance(row_id, dict):
                        existing_indices.append(row_id.get("index", 0))
            next_index = max(existing_indices, default=-1) + 1
            current_rows.append(_create_bm_row(next_index))
            return current_rows

        # Delete row
        if isinstance(triggered, dict) and triggered.get("type") == "bm-row-delete":
            delete_index = triggered["index"]
            # Check that the delete button was actually clicked
            for i, row in enumerate(current_rows):
                if isinstance(row, dict) and "props" in row:
                    row_id = row["props"].get("id", {})
                    if isinstance(row_id, dict) and row_id.get("index") == delete_index:
                        if delete_clicks and i < len(delete_clicks) and delete_clicks[i]:
                            new_rows = [r for j, r in enumerate(current_rows) if j != i]
                            return new_rows if new_rows else [_create_bm_row(0)]
                        break
            raise PreventUpdate

        raise PreventUpdate

    @app.callback(
        Output("bm-config-store", "data"),
        Output("bm-ticker-status", "children"),
        Input("apply-bm-btn", "n_clicks"),
        State({"type": "bm-row-ticker", "index": ALL}, "value"),
        State({"type": "bm-row-weight", "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def apply_bm_config(n_clicks, tickers, weights):
        """Apply benchmark config from all rows into bm-config-store."""
        if not n_clicks:
            raise PreventUpdate

        config = []
        for t, w in zip(tickers or [], weights or []):
            ticker = t.strip().upper() if t else ""
            weight = float(w) if w is not None else 0
            if ticker and weight > 0:
                config.append({"ticker": ticker, "weight": weight})

        if not config:
            return [], dbc.Alert(
                [html.I(className="fas fa-exclamation-triangle me-2"), "No valid tickers entered."],
                color="warning",
                className="py-2 mb-0",
            )

        label = " + ".join(f"{c['ticker']} {c['weight']:.0f}%" for c in config)
        return config, dbc.Alert(
            [html.I(className="fas fa-check-circle me-2"), f"Benchmark: {label}"],
            color="success",
            className="py-2 mb-0",
        )

    @app.callback(
        Output("top-k-div", "style"),
        Input("select-all-tickers-checkbox", "value"),
    )
    def toggle_top_k_slider(select_all):
        """Show/hide top-k slider based on select all checkbox."""
        if select_all:
            return {"display": "none"}
        return {"display": "block"}

    @app.callback(
        Output("random-opt-div", "style"),
        Input("opt-mode-radio", "value"),
    )
    def toggle_random_opt_settings(opt_mode):
        """Show/hide random optimization settings."""
        if opt_mode == "random":
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        Output("opt-samples-warning", "children"),
        Output("optimize-btn", "disabled"),
        Input("opt-samples-input", "value"),
        Input("opt-mode-radio", "value"),
    )
    def validate_opt_samples(samples, opt_mode):
        """Validate optimization samples and disable button if invalid."""
        # Full grid mode doesn't need samples validation
        if opt_mode == "full":
            return None, False

        if samples is None:
            return None, False

        if samples < 100:
            return dbc.Alert(
                "⚠️ Samples < 100 is too low. Minimum 100 required.",
                color="danger",
                className="py-1 px-2 mb-0",
                style={"fontSize": "0.75rem"},
            ), True

        return None, False

    @app.callback(
        Output("wf-train-warning", "children"),
        Output("wf-samples-warning", "children"),
        Output("walk-forward-btn", "disabled"),
        Input("wf-train-input", "value"),
        Input("wf-samples-input", "value"),
    )
    def validate_wf_inputs(train_months, samples):
        """Validate walk-forward inputs and disable button if invalid."""
        train_warning = None
        samples_warning = None
        disabled = False

        # Validate train period
        if train_months is not None and train_months < 12:
            train_warning = dbc.Alert(
                "⚠️ Train period must be at least 12 months.",
                color="danger",
                className="py-1 px-2 mb-0",
                style={"fontSize": "0.75rem"},
            )
            disabled = True

        # Validate samples
        if samples is not None and samples < 100:
            samples_warning = dbc.Alert(
                "⚠️ Samples < 100 is too low. Minimum 100 required.",
                color="danger",
                className="py-1 px-2 mb-0",
                style={"fontSize": "0.75rem"},
            )
            disabled = True

        return train_warning, samples_warning, disabled

    @app.callback(
        Output("wf-settings-collapse", "is_open"),
        Input("wf-settings-collapse-btn", "n_clicks"),
        State("wf-settings-collapse", "is_open"),
    )
    def toggle_wf_settings(n_clicks, is_open):
        """Toggle walk-forward settings collapse."""
        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output("signal-settings-collapse", "is_open"),
        Input("signal-settings-collapse-btn", "n_clicks"),
        State("signal-settings-collapse", "is_open"),
    )
    def toggle_signal_settings(n_clicks, is_open):
        """Toggle signal settings collapse."""
        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output("wf-results-collapse", "is_open"),
        Output("wf-results-collapse-btn", "children"),
        Input("wf-results-collapse-btn", "n_clicks"),
        State("wf-results-collapse", "is_open"),
    )
    def toggle_wf_results(n_clicks, is_open):
        """Toggle walk-forward results collapse."""
        if n_clicks:
            new_is_open = not is_open
            if new_is_open:
                btn_children = [html.I(className="fas fa-chevron-up me-1"), "Hide"]
            else:
                btn_children = [html.I(className="fas fa-chevron-down me-1"), "Show"]
            return new_is_open, btn_children
        return is_open, [html.I(className="fas fa-chevron-up me-1"), "Hide"]

    @app.callback(
        Output("user-guide-collapse", "is_open"),
        Input("user-guide-collapse-btn", "n_clicks"),
        State("user-guide-collapse", "is_open"),
    )
    def toggle_user_guide(n_clicks, is_open):
        """Toggle user guide collapse."""
        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output("raw-signal-collapse", "is_open"),
        Input("raw-signal-collapse-btn", "n_clicks"),
        State("raw-signal-collapse", "is_open"),
    )
    def toggle_raw_signal(n_clicks, is_open):
        """Toggle raw signal data collapse."""
        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output("stats-df-collapse", "is_open"),
        Input("stats-df-collapse-btn", "n_clicks"),
        State("stats-df-collapse", "is_open"),
    )
    def toggle_stats_df(n_clicks, is_open):
        """Toggle stats dataframe collapse."""
        if n_clicks:
            return not is_open
        return is_open

    # =========================================================================
    # Ticker Management Callbacks
    # =========================================================================

    @app.callback(
        Output("ticker-tags", "children"),
        Input("ticker-store", "data"),
    )
    def render_ticker_tags(tickers):
        """Render ticker tags from store."""
        if not tickers:
            return []
        tags = []
        for ticker in tickers:
            tag = dbc.Badge(
                [
                    ticker,
                    html.Span(
                        "\u00d7",
                        id={"type": "remove-ticker", "index": ticker},
                        style={"cursor": "pointer", "marginLeft": "6px", "fontWeight": "bold"},
                    ),
                ],
                color="primary",
                className="me-1 mb-1",
                style={"fontSize": "0.85rem", "padding": "5px 8px"},
            )
            tags.append(tag)
        return tags

    @app.callback(
        Output("ticker-store", "data"),
        Output("new-ticker-input", "value"),
        Input("add-ticker-btn", "n_clicks"),
        Input("new-ticker-input", "n_submit"),
        Input("reset-tickers-btn", "n_clicks"),
        Input("clear-tickers-btn", "n_clicks"),
        Input("ticker-preset-select", "value"),
        Input({"type": "remove-ticker", "index": ALL}, "n_clicks"),
        State("new-ticker-input", "value"),
        State("ticker-store", "data"),
        prevent_initial_call=True,
    )
    def manage_tickers(add_clicks, n_submit, reset_clicks, clear_clicks, preset_value, remove_clicks, new_ticker, current_tickers):
        """Add, remove, reset, clear, or load preset tickers."""
        from dash import ctx

        triggered = ctx.triggered_id

        # Preset selection - load the selected preset
        if triggered == "ticker-preset-select":
            if preset_value and preset_value in TICKER_PRESETS:
                return TICKER_PRESETS[preset_value].copy(), ""
            raise PreventUpdate

        # Reset to current preset
        if triggered == "reset-tickers-btn":
            if preset_value and preset_value in TICKER_PRESETS:
                return TICKER_PRESETS[preset_value].copy(), ""
            return TICKER_PRESETS[DEFAULT_PRESET].copy(), ""

        # Clear all tickers
        if triggered == "clear-tickers-btn":
            return [], ""

        # Remove ticker - only if actually clicked (n_clicks > 0)
        if isinstance(triggered, dict) and triggered.get("type") == "remove-ticker":
            # Find the index of this trigger in the pattern matching list
            # and check if it was actually clicked
            trigger_index = triggered["index"]
            for i, ticker in enumerate(current_tickers):
                if ticker == trigger_index:
                    if remove_clicks and i < len(remove_clicks) and remove_clicks[i]:
                        new_tickers = [t for t in current_tickers if t != trigger_index]
                        return new_tickers, dash.no_update
                    break
            raise PreventUpdate

        # Add ticker
        if triggered in ("add-ticker-btn", "new-ticker-input") and new_ticker:
            new_ticker = new_ticker.strip().upper()
            if new_ticker and new_ticker not in current_tickers:
                return current_tickers + [new_ticker], ""
            return dash.no_update, ""

        raise PreventUpdate

    # =========================================================================
    # Main Analysis Callback
    # =========================================================================

    @app.callback(
        Output("dataset-store", "data"),
        Output("bm-data-store", "data"),
        Output("params-store", "data"),
        Output("nav-store", "data"),
        Output("signal-store", "data"),
        Output("weights-store", "data"),
        Output("initial-instructions", "style"),
        Output("analysis-results", "style"),
        Output("analysis-loading", "style"),
        Output("analysis-warnings", "children"),
        Output("walk-forward-results-section", "style", allow_duplicate=True),
        Input("run-analysis-btn", "n_clicks"),
        State("start-date", "date"),
        State("backtest-start-date", "date"),
        State("ticker-store", "data"),
        State("thresh-slider", "value"),
        State("short-window-slider", "value"),
        State("mid-window-slider", "value"),
        State("long-window-slider", "value"),
        State("weight-mode-radio", "value"),
        State("short-weight-slider", "value"),
        State("mid-weight-slider", "value"),
        State("long-weight-slider", "value"),
        State("strategy-type-select", "value"),
        State("select-all-tickers-checkbox", "value"),
        State("top-k-slider", "value"),
        State("n-quantiles-slider", "value"),
        State("weight-method-select", "value"),
        State("tcost-input", "value"),
        State("bm-type-radio", "value"),
        State("bm-config-store", "data"),
        prevent_initial_call=True,
        background=True,
        running=[
            (Output("run-analysis-btn", "disabled"), True, False),
            (Output("run-analysis-btn", "children"), [dbc.Spinner(size="sm", spinner_class_name="me-2"), "Running..."], [html.I(className="fas fa-play me-2"), "Run Analysis"]),
            (Output("analysis-loading", "style"), {"display": "block"}, {"display": "none"}),
        ],
    )
    def run_analysis(
        n_clicks,
        start_date,
        backtest_start_date,
        selected_tickers,
        thresh,
        short_window,
        mid_window,
        long_window,
        weight_mode,
        short_wgt,
        mid_wgt,
        long_wgt,
        strategy_type,
        select_all,
        top_k,
        n_quantiles,
        weight_method,
        tcost,
        bm_type,
        bm_config,
    ):
        """Run the main analysis."""
        if not n_clicks or not selected_tickers or len(selected_tickers) < 2:
            raise PreventUpdate

        # Calculate weights
        if weight_mode == "equal":
            short_wgt = mid_wgt = long_wgt = 1 / 3
        else:
            total = short_wgt + mid_wgt + long_wgt
            if total > 0:
                short_wgt, mid_wgt, long_wgt = short_wgt / total, mid_wgt / total, long_wgt / total

        # Handle top_k
        if select_all:
            top_k = None

        # Ensure tcost is a number (input is in %, convert to decimal)
        tcost = float(tcost) / 100.0 if tcost else 0.0

        # Load data
        dataset, missing = load_price_data(selected_tickers, start_date=start_date)

        # Check if we have enough valid tickers
        if len(dataset.columns) < 2:
            error_msg = "Not enough valid tickers. Need at least 2 tickers with data."
            if missing:
                error_msg += f" Missing: {', '.join(missing)}"
            return (
                None, None, None, None, None, None,
                {"display": "block"},  # Show instructions
                {"display": "none"},  # Hide results
                {"display": "none"},  # Hide loading
                dbc.Alert([html.I(className="fas fa-times-circle me-2"), error_msg], color="danger"),
                {"display": "none"},  # Hide walk-forward results
            )

        # Collect warnings
        warnings = []
        if missing:
            warnings.append(
                dbc.Alert(
                    [html.I(className="fas fa-exclamation-triangle me-2"), f"Data not found for: {', '.join(missing)}"],
                    color="warning",
                    dismissable=True,
                )
            )

        # Load benchmark data if custom
        bm_data = None
        custom_bm_label = None
        if bm_type == "custom" and bm_config:
            bm_series, bm_label, bm_warns = _build_composite_bm(bm_config, start_date)
            for w in bm_warns:
                warnings.append(
                    dbc.Alert(
                        [html.I(className="fas fa-exclamation-triangle me-2"), w],
                        color="warning",
                        dismissable=True,
                    )
                )
            if bm_series is not None:
                bm_data = bm_series
                custom_bm_label = bm_label
            else:
                warnings.append(
                    dbc.Alert(
                        [html.I(className="fas fa-exclamation-triangle me-2"), "Custom BM could not be built. Using Equal Weight BM instead."],
                        color="warning",
                        dismissable=True,
                    )
                )

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

        # Run backtest
        if strategy_type == "Top-K":
            strat_nav, universe_bm_nav, turnover = run_top_k_backtest(
                dataset,
                signal,
                top_k=top_k,
                weight_method=weight_method,
                tcost=tcost,
            )
            strat_name = "All" if top_k is None else f"Top-{top_k}"
            navs = pd.DataFrame({strat_name: strat_nav})

            # Get weights for heatmap
            wgt, _ = compute_weight_top_k(dataset, signal, top_k=top_k, weight_method=weight_method)
            weights_data = {strat_name: wgt.to_json(date_format="iso")}
        else:
            q_nav, q_to, universe_bm_nav = run_quantile_backtest(
                dataset,
                signal,
                n_quantiles=n_quantiles,
                tcost=tcost,
                weight_method=weight_method,
            )
            navs = q_nav.copy()

            # Get weights for all quantiles
            q_wgts, _ = compute_weight_quantile(
                dataset, signal, n_quantiles=n_quantiles, weight_method=weight_method
            )
            weights_data = {q: wgt.to_json(date_format="iso") for q, wgt in q_wgts.items()}

        # Handle benchmark
        if custom_bm_label and bm_data is not None:
            bm_aligned = bm_data.reindex(navs.index).ffill().bfill()
            bm_nav = bm_aligned / bm_aligned.iloc[0] * 1000
            bm_nav.name = custom_bm_label
            navs["BM"] = bm_nav
        else:
            navs["BM"] = universe_bm_nav

        # Apply backtest start date
        backtest_start = pd.to_datetime(backtest_start_date)
        actual_start = navs.index[navs.index >= backtest_start]
        if len(actual_start) > 0:
            display_start = actual_start[0]
        else:
            display_start = navs.index[0]

        navs_display = navs[display_start:].copy()

        # Store params
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
            "custom_bm_label": custom_bm_label if bm_type == "custom" else None,
            "display_start": display_start.strftime("%Y-%m-%d"),
            "display_end": navs_display.index[-1].strftime("%Y-%m-%d"),
            "data_end": dataset.index[-1].strftime("%Y-%m-%d"),
        }

        # Serialize data for storage
        dataset_json = dataset.to_json(date_format="iso")
        bm_data_json = bm_data.to_json(date_format="iso") if bm_data is not None else None
        navs_json = navs_display.to_json(date_format="iso")
        signal_json = signal.to_json(date_format="iso")

        return (
            dataset_json,
            bm_data_json,
            params,
            navs_json,
            signal_json,
            weights_data,
            {"display": "none"},  # Hide instructions
            {"display": "block"},  # Show results
            {"display": "none"},  # Hide loading
            warnings if warnings else "",  # Show warnings
            {"display": "none"},  # Hide walk-forward results
        )

    # =========================================================================
    # Chart Update Callbacks
    # =========================================================================

    @app.callback(
        Output("backtest-period-info", "children"),
        Output("kpi-cards-container", "children"),
        Output("nav-chart", "figure"),
        Output("drawdown-chart", "figure"),
        Output("signal-date-info", "children"),
        Output("signal-table-chart", "figure"),
        Output("raw-signal-table-container", "children"),
        Output("quantile-spread-section", "style"),
        Output("spread-chart", "figure"),
        Output("heatmap-quantile-select", "options"),
        Output("heatmap-quantile-select", "value"),
        Output("quantile-selector-div", "style"),
        Output("heatmap-chart", "figure"),
        Output("stats-table-chart", "figure"),
        Output("stats-df-container", "children"),
        Input("nav-store", "data"),
        State("params-store", "data"),
        State("signal-store", "data"),
        State("weights-store", "data"),
    )
    def update_charts(navs_json, params, signal_json, weights_data):
        """Update all charts when NAV data changes."""
        if not navs_json or not params:
            raise PreventUpdate

        # Parse data
        navs = pd.read_json(navs_json)
        navs.index = pd.to_datetime(navs.index)
        navs = navs.sort_index()

        signal = pd.read_json(signal_json)
        signal.index = pd.to_datetime(signal.index)
        signal = signal.sort_index()

        # Rebase NAVs
        navs_rebased = rebase(navs)

        # Backtest period info
        period_info = f"Backtest Period: {params['display_start']} ~ {params['display_end']}"

        # KPI Cards
        kpis = calculate_kpis(navs)
        kpi_cards = []
        for name, metrics in kpis.items():
            card = dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5(name, className="card-title"),
                            html.P(f"CAGR: {metrics['cagr']:.1%}", className="mb-1"),
                            html.P(f"Vol: {metrics['vol']:.1%}", className="mb-1"),
                            html.P(f"Sharpe: {metrics['sharpe']:.2f}", className="mb-1"),
                            html.P(f"MDD: {metrics['mdd']:.1%}", className="mb-1"),
                            html.P(f"YTD: {metrics['ytd']:.1%}", className="mb-0"),
                        ]
                    ),
                    className="h-100",
                ),
                width=True,
                className="mb-3",
            )
            kpi_cards.append(card)
        kpi_row = dbc.Row(kpi_cards)

        # NAV Chart
        fig_nav = create_nav_chart(navs_rebased, title="Portfolio NAV (Rebased to 100)")

        # Drawdown Chart
        fig_dd = create_drawdown_chart(navs_rebased, title="Drawdown (%)", height=350)

        # Signal Category Table
        latest_signal = get_signal_ranking(signal)
        signal_date_info = f"Date: {params['data_end']}"
        fig_category = create_signal_category_table(
            latest_signal, n_quantiles=params["n_quantiles"], title=""
        )

        # Raw Signal Data
        signal_display = signal.iloc[-5:].T.copy()
        signal_display.columns = signal_display.columns.strftime("%Y-%m-%d")
        raw_signal_table = dash_table.DataTable(
            data=signal_display.reset_index().to_dict("records"),
            columns=[{"name": col, "id": col} for col in ["index"] + list(signal_display.columns)],
            style_cell={"textAlign": "center"},
            style_header={"fontWeight": "bold"},
        )

        # Quantile Spread (only for Quantile strategy)
        if params["strategy_type"] == "Quantile" and "Q1" in navs.columns and "Q5" in navs.columns:
            spread_style = {"display": "block"}
            fig_spread = create_quantile_spread_chart(navs)
        else:
            spread_style = {"display": "none"}
            fig_spread = go.Figure()

        # Heatmap quantile selector
        if params["strategy_type"] == "Quantile":
            n_q = params["n_quantiles"]
            quantile_options = [
                {"label": f"Q{q}", "value": f"Q{q}"} for q in range(n_q, 0, -1)
            ]
            default_quantile = f"Q{n_q}"
            quantile_selector_style = {"display": "block"}
        else:
            strat_name = "All" if params["top_k"] is None else f"Top-{params['top_k']}"
            quantile_options = [{"label": strat_name, "value": strat_name}]
            default_quantile = strat_name
            quantile_selector_style = {"display": "none"}

        # Heatmap
        wgt_json = weights_data.get(default_quantile)
        if wgt_json:
            wgt = pd.read_json(wgt_json)
            wgt.index = pd.to_datetime(wgt.index)
            wgt = wgt.sort_index()
            wgt.index = wgt.index + BMonthBegin(1)
            recent_wgt = wgt.iloc[-36:]
            n_tickers = len(recent_wgt.columns)
            chart_height = max(500, n_tickers * 25)
            fig_heatmap = create_holding_heatmap(
                recent_wgt,
                title=f"Monthly Holdings - {default_quantile} (Last 36 Months)",
                height=chart_height,
            )
        else:
            fig_heatmap = go.Figure()

        # Stats Table
        stats = summary_stats(navs)
        fig_stats = create_returns_table(stats)

        # Stats DataFrame (formatted like the table)
        formatted_stats = stats.copy()
        pct_rows = ['cumulative', 'cagr', 'mean', 'vol', 'max', 'min', 'mdd']
        for row in pct_rows:
            if row in formatted_stats.index:
                formatted_stats.loc[row] = formatted_stats.loc[row].apply(lambda x: f'{x:.1%}')
        ratio_rows = ['sharpe', 'skew', 'kurt']
        for row in ratio_rows:
            if row in formatted_stats.index:
                formatted_stats.loc[row] = formatted_stats.loc[row].apply(lambda x: f'{x:.2f}')
        if 'nyears' in formatted_stats.index:
            formatted_stats.loc['nyears'] = formatted_stats.loc['nyears'].apply(lambda x: f'{x:.1f}')
        if 'nsamples' in formatted_stats.index:
            formatted_stats.loc['nsamples'] = formatted_stats.loc['nsamples'].apply(lambda x: f'{int(x)}')

        stats_df = formatted_stats.T.reset_index()
        stats_df_table = dash_table.DataTable(
            data=stats_df.to_dict("records"),
            columns=[{"name": str(col), "id": str(col)} for col in stats_df.columns],
            style_cell={"textAlign": "center"},
            style_header={"fontWeight": "bold"},
        )

        return (
            period_info,
            kpi_row,
            fig_nav,
            fig_dd,
            signal_date_info,
            fig_category,
            raw_signal_table,
            spread_style,
            fig_spread,
            quantile_options,
            default_quantile,
            quantile_selector_style,
            fig_heatmap,
            fig_stats,
            stats_df_table,
        )

    @app.callback(
        Output("heatmap-chart", "figure", allow_duplicate=True),
        Output("quantile-info-div", "children"),
        Input("heatmap-quantile-select", "value"),
        State("weights-store", "data"),
        State("params-store", "data"),
        prevent_initial_call=True,
    )
    def update_heatmap_quantile(selected_quantile, weights_data, params):
        """Update heatmap when quantile selection changes."""
        if not selected_quantile or not weights_data or not params:
            raise PreventUpdate

        wgt_json = weights_data.get(selected_quantile)
        if not wgt_json:
            raise PreventUpdate

        wgt = pd.read_json(wgt_json)
        wgt.index = pd.to_datetime(wgt.index)
        wgt = wgt.sort_index()
        wgt.index = wgt.index + BMonthBegin(1)
        recent_wgt = wgt.iloc[-36:]
        n_tickers = len(recent_wgt.columns)
        chart_height = max(500, n_tickers * 25)

        fig_heatmap = create_holding_heatmap(
            recent_wgt,
            title=f"Monthly Holdings - {selected_quantile} (Last 36 Months)",
            height=chart_height,
        )

        # Quantile info
        if params["strategy_type"] == "Quantile":
            n_q = params["n_quantiles"]
            if selected_quantile == f"Q{n_q}":
                desc = "Best performers (lowest ranks)"
            elif selected_quantile == "Q1":
                desc = "Worst performers (highest ranks)"
            else:
                desc = "Middle performers"
            info = dbc.Alert(f"{selected_quantile}: {desc}", color="info", className="mb-0 py-2")
        else:
            info = None

        return fig_heatmap, info

    # =========================================================================
    # Optimization Callback (Background Callback)
    # =========================================================================

    @app.callback(
        Output("optimized-params-store", "data"),
        Output("optimize-result-div", "children"),
        Output("optimize-progress-div", "children", allow_duplicate=True),
        Input("optimize-btn", "n_clicks"),
        State("start-date", "date"),
        State("backtest-start-date", "date"),
        State("ticker-store", "data"),
        State("thresh-slider", "value"),
        State("select-all-tickers-checkbox", "value"),
        State("top-k-slider", "value"),
        State("weight-method-select", "value"),
        State("tcost-input", "value"),
        State("opt-mode-radio", "value"),
        State("opt-samples-input", "value"),
        State("opt-seed-input", "value"),
        background=True,
        running=[
            (Output("optimize-progress-div", "children"), dbc.Progress(value=0, striped=True, animated=True), ""),
        ],
        progress=[Output("optimize-progress-div", "children")],
        prevent_initial_call=True,
    )
    def run_optimization(
        set_progress,
        n_clicks,
        start_date,
        backtest_start_date,
        selected_tickers,
        thresh,
        select_all,
        top_k,
        weight_method,
        tcost,
        opt_mode,
        n_samples,
        opt_seed,
    ):
        """Run parameter optimization."""
        if not n_clicks or not selected_tickers or len(selected_tickers) < 2:
            raise PreventUpdate

        # Validate samples
        if opt_mode == "random" and n_samples:
            n_samples = int(n_samples)
            if n_samples < 100:
                return None, dbc.Alert("Samples must be at least 100.", color="danger"), ""
        else:
            n_samples = None

        # Load data
        dataset, _ = load_price_data(selected_tickers, start_date=start_date)

        # Handle parameters
        if select_all:
            top_k = None
        tcost = float(tcost) / 100.0 if tcost else 0.0
        # Parse seed from text input
        try:
            opt_seed = int(opt_seed) if opt_seed and str(opt_seed).strip().isdigit() else None
        except (ValueError, TypeError):
            opt_seed = None

        # Progress callback
        def update_progress(current, total):
            progress = int(current / total * 100)
            set_progress(dbc.Progress(value=progress, label=f"{current}/{total}", striped=True, animated=True))

        # Run optimization
        result = optimize_sharpe(
            price=dataset,
            top_k=top_k,
            weight_method=weight_method,
            tcost=tcost,
            start_date=str(backtest_start_date),
            thresh=thresh,
            n_samples=n_samples,
            seed=opt_seed,
            progress_callback=update_progress,
        )

        if result["best_params"] is None:
            return None, dbc.Alert("Optimization failed. Try different settings.", color="danger"), ""

        bp = result["best_params"]
        tested_info = f"{result['total_combinations']}/{result['total_possible']}"

        result_div = dbc.Alert(
            [
                html.Strong("Optimization Complete!"),
                html.Span(f" (tested {tested_info})"),
                html.Br(),
                html.Span(f"Sharpe: {result['best_sharpe']:.3f}"),
                html.Br(),
                html.Span(f"Windows: {bp['short_window']}/{bp['mid_window']}/{bp['long_window']} mo"),
                html.Br(),
                html.Span(f"Weights: {bp['short_wgt']:.0%}/{bp['mid_wgt']:.0%}/{bp['long_wgt']:.0%}"),
                html.Br(),
                html.Small("Click 'Run Analysis' to apply parameters."),
            ],
            color="success",
        )

        return bp, result_div, ""

    @app.callback(
        Output("short-window-slider", "value"),
        Output("mid-window-slider", "value"),
        Output("long-window-slider", "value"),
        Output("short-weight-slider", "value"),
        Output("mid-weight-slider", "value"),
        Output("long-weight-slider", "value"),
        Output("weight-mode-radio", "value"),
        Input("optimized-params-store", "data"),
        prevent_initial_call=True,
    )
    def apply_optimized_params(optimized_params):
        """Apply optimized parameters to sliders."""
        if not optimized_params:
            raise PreventUpdate

        return (
            optimized_params["short_window"],
            optimized_params["mid_window"],
            optimized_params["long_window"],
            optimized_params["short_wgt"],
            optimized_params["mid_wgt"],
            optimized_params["long_wgt"],
            "custom",
        )

    # =========================================================================
    # Walk-Forward Callback (Background Callback)
    # =========================================================================

    @app.callback(
        Output("walk-forward-store", "data"),
        Output("wf-result-div", "children"),
        Output("wf-progress-div", "children", allow_duplicate=True),
        # Walk-forward results section outputs (directly updated from background callback)
        Output("walk-forward-results-section", "style", allow_duplicate=True),
        Output("wf-results-collapse", "is_open", allow_duplicate=True),
        Output("wf-results-collapse-btn", "children", allow_duplicate=True),
        Output("wf-results-title", "children"),
        Output("wf-is-sharpe", "children"),
        Output("wf-oos-sharpe", "children"),
        Output("wf-combined-sharpe", "children"),
        Output("wf-sharpe-decay", "children"),
        Output("wf-interpretation-div", "children"),
        Output("wf-fold-table-container", "children"),
        Output("wf-train-period-info", "children"),
        Output("wf-rec-windows", "children"),
        Output("wf-rec-weights", "children"),
        Output("wf-rec-sharpe", "children"),
        Output("wf-stability-table-container", "children"),
        Output("wf-oos-nav-info", "children"),
        Output("wf-oos-nav-chart", "figure"),
        Output("wf-oos-sharpe-comparison", "children"),
        Input("walk-forward-btn", "n_clicks"),
        State("start-date", "date"),
        State("ticker-store", "data"),
        State("thresh-slider", "value"),
        State("select-all-tickers-checkbox", "value"),
        State("top-k-slider", "value"),
        State("weight-method-select", "value"),
        State("tcost-input", "value"),
        State("wf-window-type-radio", "value"),
        State("wf-train-input", "value"),
        State("wf-test-input", "value"),
        State("wf-step-input", "value"),
        State("wf-samples-input", "value"),
        State("wf-seed-input", "value"),
        State("bm-type-radio", "value"),
        State("bm-config-store", "data"),
        background=True,
        running=[
            (Output("wf-progress-div", "children"), dbc.Progress(value=0, striped=True, animated=True), ""),
        ],
        progress=[Output("wf-progress-div", "children")],
        prevent_initial_call=True,
    )
    def run_walk_forward(
        set_progress,
        n_clicks,
        start_date,
        selected_tickers,
        thresh,
        select_all,
        top_k,
        weight_method,
        tcost,
        window_type,
        train_months,
        test_months,
        step_months,
        wf_samples,
        wf_seed,
        bm_type,
        bm_config,
    ):
        """Run walk-forward optimization."""
        # Default empty results for error cases
        empty_results = (
            {"display": "none"},  # section style
            False,  # collapse is_open
            "",  # collapse btn children
            "",  # title
            "",  # is_sharpe
            "",  # oos_sharpe
            "",  # combined_sharpe
            "",  # decay
            "",  # interpretation
            "",  # fold_table
            "",  # train_period_info
            "",  # rec_windows
            "",  # rec_weights
            "",  # rec_sharpe
            "",  # stability_table
            "",  # oos_nav_info
            go.Figure(),  # oos_nav_chart
            "",  # sharpe_comparison
        )

        if not n_clicks or not selected_tickers or len(selected_tickers) < 2:
            raise PreventUpdate

        # Validate train period
        train_months = int(train_months) if train_months else 36
        if train_months < 12:
            return (None, dbc.Alert("Train period must be at least 12 months.", color="danger"), "") + empty_results

        # Validate samples
        wf_samples = int(wf_samples) if wf_samples else 500
        if wf_samples < 100:
            return (None, dbc.Alert("Samples must be at least 100.", color="danger"), "") + empty_results

        # Load data
        dataset, _ = load_price_data(selected_tickers, start_date=start_date)

        # Load benchmark data if custom
        bm_data = None
        wf_bm_label = None
        if bm_type == "custom" and bm_config:
            bm_series, bm_label, _ = _build_composite_bm(bm_config, start_date)
            if bm_series is not None:
                bm_data = bm_series
                wf_bm_label = bm_label

        # Handle parameters
        if select_all:
            top_k = None
        tcost = float(tcost) / 100.0 if tcost else 0.0
        # Parse seed from text input
        try:
            wf_seed = int(wf_seed) if wf_seed and str(wf_seed).strip().isdigit() else None
        except (ValueError, TypeError):
            wf_seed = None

        # Progress callback
        def update_progress(fold, total_folds, step, total_steps):
            if fold > total_folds:
                overall = int((total_folds * total_steps + step) / ((total_folds + 1) * total_steps) * 100)
                label = f"Final Training - {step}/{total_steps}"
            else:
                overall = int(((fold - 1) * total_steps + step) / ((total_folds + 1) * total_steps) * 100)
                label = f"Fold {fold}/{total_folds} - {step}/{total_steps}"
            set_progress(dbc.Progress(value=overall, label=label, striped=True, animated=True))

        # Run walk-forward
        wf_result = walk_forward_optimize(
            price=dataset,
            train_months=int(train_months),
            test_months=int(test_months),
            step_months=int(step_months),
            window_type=window_type,
            top_k=top_k,
            weight_method=weight_method,
            tcost=tcost,
            thresh=thresh,
            n_samples=wf_samples,
            seed=wf_seed,
            bm_ticker=wf_bm_label,
            bm_data=bm_data,
            progress_callback=update_progress,
        )

        if "error" in wf_result:
            return (None, dbc.Alert(wf_result["error"], color="danger"), "") + empty_results

        # Store result (convert Series to JSON)
        wf_store = {
            "folds": wf_result["folds"],
            "oos_sharpe": wf_result["oos_sharpe"],
            "oos_sharpe_avg": wf_result["oos_sharpe_avg"],
            "is_sharpe_avg": wf_result["is_sharpe_avg"],
            "sharpe_decay": wf_result["sharpe_decay"],
            "param_stability": wf_result["param_stability"],
            "total_folds": wf_result["total_folds"],
            "valid_folds": wf_result["valid_folds"],
            "window_type": wf_result["window_type"],
            "final_params": wf_result["final_params"],
            "final_sharpe": wf_result["final_sharpe"],
            "final_train_period": wf_result["final_train_period"],
            "oos_date_range": wf_result["oos_date_range"],
            "combined_oos_nav": wf_result["combined_oos_nav"].to_json(date_format="iso") if wf_result["combined_oos_nav"] is not None else None,
            "combined_oos_bm_nav": wf_result["combined_oos_bm_nav"].to_json(date_format="iso") if wf_result["combined_oos_bm_nav"] is not None else None,
            "_bm_type": bm_type,
            "_bm_ticker": wf_bm_label if bm_type == "custom" else None,
        }

        decay = wf_result["sharpe_decay"]
        decay_str = f"{decay:+.2f}" if not np.isnan(decay) else "N/A"
        wf_type = wf_result["window_type"].capitalize()

        result_div = dbc.Alert(
            [
                html.Strong(f"Walk-Forward Complete!"),
                html.Span(f" ({wf_result['valid_folds']}/{wf_result['total_folds']} folds, {wf_type})"),
                html.Br(),
                html.Span(f"IS Sharpe (avg): {wf_result['is_sharpe_avg']:.2f}"),
                html.Br(),
                html.Span(f"OOS Sharpe (avg): {wf_result['oos_sharpe_avg']:.2f}"),
                html.Br(),
                html.Span(f"Combined OOS Sharpe: {wf_result['oos_sharpe']:.2f}"),
                html.Br(),
                html.Span(f"Sharpe Decay: {decay_str}"),
            ],
            color="info",
        )

        # ===== Build results section UI directly =====
        # Title
        title = f"Walk-Forward Analysis ({wf_type} Window)"

        # Metrics
        is_sharpe_str = f"{wf_store['is_sharpe_avg']:.2f}"
        oos_sharpe_str = f"{wf_store['oos_sharpe_avg']:.2f}"
        combined_sharpe_str = f"{wf_store['oos_sharpe']:.2f}"
        decay_display = f"{decay:.2f}" if not np.isnan(decay) else "N/A"

        # Interpretation
        if not np.isnan(decay):
            if decay > 0.5:
                interpretation = dbc.Alert(
                    "High Sharpe Decay: In-sample performance significantly exceeds out-of-sample. This suggests potential overfitting.",
                    color="warning",
                )
            elif decay > 0.2:
                interpretation = dbc.Alert(
                    "Moderate Sharpe Decay: Some performance degradation out-of-sample. Consider simpler parameter choices.",
                    color="info",
                )
            else:
                interpretation = dbc.Alert(
                    "Low Sharpe Decay: Strategy shows reasonable stability between in-sample and out-of-sample performance.",
                    color="success",
                )
        else:
            interpretation = None

        # Fold table
        fold_data = []
        for f in wf_store["folds"]:
            fold_data.append(
                {
                    "Fold": f["fold"],
                    "Train Period": f"{f['train_start'][:7]} ~ {f['train_end'][:7]}",
                    "Test Period": f"{f['test_start'][:7]} ~ {f['test_end'][:7]}",
                    "Windows": f"{f['params']['short_window']}/{f['params']['mid_window']}/{f['params']['long_window']}",
                    "Weights": f"{f['params']['short_wgt']:.0%}/{f['params']['mid_wgt']:.0%}/{f['params']['long_wgt']:.0%}",
                    "IS Sharpe": f"{f['is_sharpe']:.2f}",
                    "OOS Sharpe": f"{f['oos_sharpe']:.2f}",
                }
            )
        fold_table = dash_table.DataTable(
            data=fold_data,
            columns=[{"name": col, "id": col} for col in fold_data[0].keys()] if fold_data else [],
            style_cell={"textAlign": "center"},
            style_header={"fontWeight": "bold"},
        )

        # Recommended params
        fp = wf_store["final_params"]
        ftp = wf_store["final_train_period"]
        train_period_info = f"Trained on latest data: {ftp['start'][:7]} ~ {ftp['end'][:7]}"
        rec_windows = f"{fp['short_window']}/{fp['mid_window']}/{fp['long_window']} mo"
        rec_weights = f"{fp['short_wgt']:.0%}/{fp['mid_wgt']:.0%}/{fp['long_wgt']:.0%}"
        rec_sharpe = f"{wf_store['final_sharpe']:.2f}"

        # Parameter stability table
        stability = wf_store["param_stability"]
        if stability:
            stability_data = []
            labels = {
                "short_window": "Short Window",
                "mid_window": "Mid Window",
                "long_window": "Long Window",
                "short_wgt": "Short Weight",
                "mid_wgt": "Mid Weight",
                "long_wgt": "Long Weight",
            }
            for key, label in labels.items():
                if key in stability:
                    s = stability[key]
                    multiplier = 100 if "wgt" in key else 1
                    stability_data.append(
                        {
                            "Parameter": label,
                            "Mean": f"{s['mean'] * multiplier:.1f}",
                            "Std": f"{s['std'] * multiplier:.1f}",
                            "Min": f"{s['min'] * multiplier:.1f}",
                            "Max": f"{s['max'] * multiplier:.1f}",
                        }
                    )
            stability_table = dash_table.DataTable(
                data=stability_data,
                columns=[{"name": col, "id": col} for col in stability_data[0].keys()] if stability_data else [],
                style_cell={"textAlign": "center"},
                style_header={"fontWeight": "bold"},
            )
        else:
            stability_table = html.P("Not enough folds for stability analysis.")

        # OOS NAV chart
        bm_name = wf_store["_bm_ticker"] if wf_store["_bm_type"] == "custom" and wf_store["_bm_ticker"] else "Equal Weight"
        oos_nav_info = f"NAV from chaining all out-of-sample test periods. BM = {bm_name}"

        fig_oos = go.Figure()
        sharpe_comparison = ""

        if wf_store["combined_oos_nav"]:
            oos_nav = pd.read_json(wf_store["combined_oos_nav"], typ="series")
            oos_nav.index = pd.to_datetime(oos_nav.index)
            oos_nav = oos_nav.sort_index()
            oos_nav_rebased = oos_nav / oos_nav.iloc[0] * 100

            fig_oos.add_trace(
                go.Scatter(
                    x=oos_nav_rebased.index,
                    y=oos_nav_rebased.values,
                    mode="lines",
                    name="Strategy (OOS)",
                    line=dict(color="#2196F3", width=2),
                )
            )

            # Use stored benchmark NAV from walk-forward optimization
            oos_bm_nav = None
            if wf_store.get("combined_oos_bm_nav"):
                oos_bm_nav = pd.read_json(wf_store["combined_oos_bm_nav"], typ="series")
                oos_bm_nav.index = pd.to_datetime(oos_bm_nav.index)
                oos_bm_nav = oos_bm_nav.sort_index()
                oos_bm_nav = oos_bm_nav / oos_bm_nav.iloc[0] * 100

            if oos_bm_nav is not None and len(oos_bm_nav) > 0:
                fig_oos.add_trace(
                    go.Scatter(
                        x=oos_bm_nav.index,
                        y=oos_bm_nav.values,
                        mode="lines",
                        name=f"BM ({bm_name})",
                        line=dict(color="#9E9E9E", width=1.5, dash="dash"),
                    )
                )

            fig_oos.update_layout(
                title="Walk-Forward Out-of-Sample Performance",
                xaxis_title="Date",
                yaxis_title="NAV",
                template="plotly_white",
                height=400,
                dragmode=False,
                xaxis=dict(fixedrange=True),
                yaxis=dict(fixedrange=True),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            # Sharpe comparison
            oos_sharpe_val = wf_store["oos_sharpe"]
            if oos_bm_nav is not None and len(oos_bm_nav) > 1:
                bm_daily_ret = oos_bm_nav.pct_change().dropna()
                if len(bm_daily_ret) > 0 and bm_daily_ret.std() > 0:
                    oos_bm_sharpe = bm_daily_ret.mean() / bm_daily_ret.std() * np.sqrt(252)
                    excess = oos_sharpe_val - oos_bm_sharpe
                    sharpe_comparison = f"OOS Sharpe: Strategy {oos_sharpe_val:.2f} vs BM {oos_bm_sharpe:.2f} (Excess: {excess:+.2f})"

        return (
            wf_store,
            result_div,
            "",  # wf-progress-div
            {"display": "block"},  # walk-forward-results-section style
            True,  # wf-results-collapse is_open
            [html.I(className="fas fa-chevron-up me-1"), "Hide"],  # wf-results-collapse-btn
            title,  # wf-results-title
            is_sharpe_str,  # wf-is-sharpe
            oos_sharpe_str,  # wf-oos-sharpe
            combined_sharpe_str,  # wf-combined-sharpe
            decay_display,  # wf-sharpe-decay
            interpretation,  # wf-interpretation-div
            fold_table,  # wf-fold-table-container
            train_period_info,  # wf-train-period-info
            rec_windows,  # wf-rec-windows
            rec_weights,  # wf-rec-weights
            rec_sharpe,  # wf-rec-sharpe
            stability_table,  # wf-stability-table-container
            oos_nav_info,  # wf-oos-nav-info
            fig_oos,  # wf-oos-nav-chart
            sharpe_comparison,  # wf-oos-sharpe-comparison
        )

    @app.callback(
        Output("short-window-slider", "value", allow_duplicate=True),
        Output("mid-window-slider", "value", allow_duplicate=True),
        Output("long-window-slider", "value", allow_duplicate=True),
        Output("short-weight-slider", "value", allow_duplicate=True),
        Output("mid-weight-slider", "value", allow_duplicate=True),
        Output("long-weight-slider", "value", allow_duplicate=True),
        Output("weight-mode-radio", "value", allow_duplicate=True),
        Input("apply-wf-params-btn", "n_clicks"),
        State("walk-forward-store", "data"),
        prevent_initial_call=True,
    )
    def apply_wf_params(n_clicks, wf_store):
        """Apply walk-forward recommended parameters."""
        if not n_clicks or not wf_store or not wf_store.get("final_params"):
            raise PreventUpdate

        fp = wf_store["final_params"]

        return (
            fp["short_window"],
            fp["mid_window"],
            fp["long_window"],
            fp["short_wgt"],
            fp["mid_wgt"],
            fp["long_wgt"],
            "custom",
        )
