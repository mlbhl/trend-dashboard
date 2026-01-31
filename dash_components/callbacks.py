"""Callback functions for Trend Dashboard Dash app."""

import json
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, html, no_update, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from pandas.tseries.offsets import BMonthBegin

from src.config import ALPHA_LIST, WEIGHT_METHODS
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


def register_callbacks(app):
    """Register all callbacks for the Dash app."""

    # =========================================================================
    # UI Toggle Callbacks
    # =========================================================================

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
        Output("bm-ticker-status", "children"),
        Input("bm-ticker-input", "value"),
        Input("apply-bm-ticker-btn", "n_clicks"),
        State("bm-ticker-input", "value"),
        prevent_initial_call=True,
    )
    def update_bm_ticker_status(input_value, n_clicks, ticker_value):
        """Show the currently set benchmark ticker."""
        from dash import ctx

        ticker = ticker_value.strip().upper() if ticker_value else ""
        if ticker:
            return dbc.Alert(
                [html.I(className="fas fa-check-circle me-2"), f"Benchmark: {ticker}"],
                color="success",
                className="py-2 mb-0",
            )
        return ""

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

        if samples < 200:
            return dbc.Alert(
                "⚠️ Samples < 200 may produce unreliable results.",
                color="warning",
                className="py-1 px-2 mb-0",
                style={"fontSize": "0.75rem"},
            ), False

        return None, False

    @app.callback(
        Output("wf-samples-warning", "children"),
        Input("wf-samples-input", "value"),
    )
    def show_wf_samples_warning(samples):
        """Show warning if walk-forward samples is too low."""
        if samples is None:
            return None
        if samples < 100:
            return dbc.Alert(
                "⚠️ Samples < 100 is too low. Minimum 100 required.",
                color="danger",
                className="py-1 px-2 mb-0",
                style={"fontSize": "0.75rem"},
            )
        if samples < 200:
            return dbc.Alert(
                "⚠️ Samples < 200 may produce unreliable results.",
                color="warning",
                className="py-1 px-2 mb-0",
                style={"fontSize": "0.75rem"},
            )
        return None

    @app.callback(
        Output("wf-train-warning", "children"),
        Output("walk-forward-btn", "disabled"),
        Input("wf-train-input", "value"),
        Input("wf-samples-input", "value"),
    )
    def validate_wf_inputs(train_months, samples):
        """Validate walk-forward inputs and disable button if invalid."""
        warning = None
        disabled = False

        # Validate train period
        if train_months is not None and train_months < 12:
            warning = dbc.Alert(
                "⚠️ Train period must be at least 12 months.",
                color="danger",
                className="py-1 px-2 mb-0",
                style={"fontSize": "0.75rem"},
            )
            disabled = True
        elif train_months is not None and train_months > 60:
            warning = dbc.Alert(
                "⚠️ Train > 60 months may leave insufficient test data.",
                color="warning",
                className="py-1 px-2 mb-0",
                style={"fontSize": "0.75rem"},
            )

        # Validate samples (only block, don't override warning)
        if samples is not None and samples < 100:
            disabled = True

        return warning, disabled

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
        Output("ticker-select", "options"),
        Output("ticker-select", "value"),
        Input("add-ticker-btn", "n_clicks"),
        Input("reset-tickers-btn", "n_clicks"),
        State("new-ticker-input", "value"),
        State("ticker-select", "options"),
        State("ticker-select", "value"),
        prevent_initial_call=True,
    )
    def manage_tickers(add_clicks, reset_clicks, new_ticker, current_options, current_value):
        """Add new ticker or reset to defaults."""
        from dash import ctx

        triggered_id = ctx.triggered_id

        if triggered_id == "reset-tickers-btn":
            default_options = [{"label": t, "value": t} for t in ALPHA_LIST]
            return default_options, ALPHA_LIST.copy()

        if triggered_id == "add-ticker-btn" and new_ticker:
            new_ticker = new_ticker.strip().upper()
            existing_values = [opt["value"] for opt in current_options]

            if new_ticker and new_ticker not in existing_values:
                new_options = current_options + [{"label": new_ticker, "value": new_ticker}]
                new_value = current_value + [new_ticker] if current_value else [new_ticker]
                return new_options, new_value

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
        Input("run-analysis-btn", "n_clicks"),
        State("start-date", "date"),
        State("backtest-start-date", "date"),
        State("ticker-select", "value"),
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
        State("bm-ticker-input", "value"),
        prevent_initial_call=True,
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
        bm_ticker,
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

        # Ensure tcost is a number
        tcost = float(tcost) if tcost else 0.0

        # Load data
        dataset, missing = load_price_data(selected_tickers, start_date=start_date)

        # Load benchmark data if custom
        bm_data = None
        custom_bm_ticker = None
        if bm_type == "custom" and bm_ticker:
            custom_bm_ticker = bm_ticker.strip().upper()
            if custom_bm_ticker not in dataset.columns:
                bm_dataset, _ = load_price_data([custom_bm_ticker], start_date=start_date)
                if custom_bm_ticker in bm_dataset.columns:
                    bm_data = bm_dataset[custom_bm_ticker]

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
        if custom_bm_ticker and bm_data is not None:
            bm_aligned = bm_data.reindex(navs.index).ffill().bfill()
            bm_nav = bm_aligned / bm_aligned.iloc[0] * 1000
            bm_nav.name = custom_bm_ticker
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
            "custom_bm_ticker": custom_bm_ticker if bm_type == "custom" else None,
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

        # Stats DataFrame
        stats_df_table = dash_table.DataTable(
            data=stats.T.reset_index().to_dict("records"),
            columns=[{"name": col, "id": col} for col in ["index"] + list(stats.columns)],
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
        State("ticker-select", "value"),
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
        tcost = float(tcost) if tcost else 0.0
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
        Input("walk-forward-btn", "n_clicks"),
        State("start-date", "date"),
        State("ticker-select", "value"),
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
        State("bm-ticker-input", "value"),
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
        bm_ticker,
    ):
        """Run walk-forward optimization."""
        if not n_clicks or not selected_tickers or len(selected_tickers) < 2:
            raise PreventUpdate

        # Validate train period
        train_months = int(train_months) if train_months else 36
        if train_months < 12:
            return None, dbc.Alert("Train period must be at least 12 months.", color="danger"), ""

        # Validate samples
        wf_samples = int(wf_samples) if wf_samples else 500
        if wf_samples < 100:
            return None, dbc.Alert("Samples must be at least 100.", color="danger"), ""

        # Load data
        dataset, _ = load_price_data(selected_tickers, start_date=start_date)

        # Load benchmark data if custom
        bm_data = None
        wf_bm_ticker = None
        if bm_type == "custom" and bm_ticker:
            wf_bm_ticker = bm_ticker.strip().upper()
            if wf_bm_ticker not in dataset.columns:
                bm_dataset, _ = load_price_data([wf_bm_ticker], start_date=start_date)
                if wf_bm_ticker in bm_dataset.columns:
                    bm_data = bm_dataset[wf_bm_ticker]

        # Handle parameters
        if select_all:
            top_k = None
        tcost = float(tcost) if tcost else 0.0
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
            bm_ticker=wf_bm_ticker,
            bm_data=bm_data,
            progress_callback=update_progress,
        )

        if "error" in wf_result:
            return None, dbc.Alert(wf_result["error"], color="danger"), ""

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
            "_bm_type": bm_type,
            "_bm_ticker": wf_bm_ticker if bm_type == "custom" else None,
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

        return wf_store, result_div, ""

    @app.callback(
        Output("walk-forward-results-section", "style"),
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
        Input("walk-forward-store", "data"),
        State("dataset-store", "data"),
        prevent_initial_call=True,
    )
    def update_walk_forward_results(wf_store, dataset_json):
        """Update walk-forward results display."""
        if not wf_store:
            raise PreventUpdate

        # Title
        wf_type = wf_store["window_type"].capitalize()
        title = f"Walk-Forward Analysis ({wf_type} Window)"

        # Metrics
        is_sharpe = f"{wf_store['is_sharpe_avg']:.2f}"
        oos_sharpe = f"{wf_store['oos_sharpe_avg']:.2f}"
        combined_sharpe = f"{wf_store['oos_sharpe']:.2f}"

        decay = wf_store["sharpe_decay"]
        decay_str = f"{decay:.2f}" if not np.isnan(decay) else "N/A"

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

            # Compute benchmark NAV if dataset available
            oos_bm_nav = None
            oos_date_range = wf_store.get("oos_date_range")

            if oos_date_range and dataset_json:
                dataset = pd.read_json(dataset_json)
                dataset.index = pd.to_datetime(dataset.index)
                dataset = dataset.sort_index()

                oos_start = oos_date_range["start"]
                oos_end = oos_date_range["end"]

                # Equal weight benchmark
                dataset_slice = dataset[oos_start:oos_end]
                if len(dataset_slice) > 0:
                    daily_ret = dataset_slice.pct_change().fillna(0)
                    ew_ret = daily_ret.mean(axis=1)
                    oos_bm_nav = (1 + ew_ret).cumprod() * 100

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
                else:
                    sharpe_comparison = ""
            else:
                sharpe_comparison = ""
        else:
            sharpe_comparison = ""

        return (
            {"display": "block"},
            title,
            is_sharpe,
            oos_sharpe,
            combined_sharpe,
            decay_str,
            interpretation,
            fold_table,
            train_period_info,
            rec_windows,
            rec_weights,
            rec_sharpe,
            stability_table,
            oos_nav_info,
            fig_oos,
            sharpe_comparison,
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
