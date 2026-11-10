"""Optimization callbacks: parameter grid search (Optimize) and walk-forward."""

from io import StringIO

from dash import Input, Output, State, html, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data import load_price_data
from src.metrics import sharpe_from_nav
from src.optimizer import optimize_sharpe, walk_forward_optimize
from src.analysis import (
    AnalysisError,
    build_composite_bm,
    check_backtest_dates,
    check_tcost,
    check_whole_number,
    parse_core_config,
    parse_seed,
    with_core_prices,
)


def _enough_tickers(tickers, thresh) -> bool:
    return len(tickers or []) >= (thresh or 0)


def _wf_input_problems(train_months, test_months, samples) -> tuple:
    return (
        check_whole_number(train_months, 12, "Train period (months)"),
        check_whole_number(test_months, 1, "Test period (months)"),
        check_whole_number(samples, 100, "Samples per fold"),
    )


def _input_alert(problem: str | None):
    if not problem:
        return None
    return dbc.Alert(f"⚠️ {problem}", color="danger", className="py-1 px-2 mb-0", style={"fontSize": "0.75rem"})


def _error_alert(err) -> dbc.Alert:
    return dbc.Alert([html.I(className="fas fa-times-circle me-2"), str(err)], color="danger")


def register(app):
    """Register optimization callbacks."""

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
        Input("ticker-store", "data"),
        Input("thresh-slider", "value"),
    )
    def validate_opt_inputs(samples, opt_mode, tickers, thresh):
        """Only writer of optimize-btn.disabled: sample count (random mode) and ticker count."""
        problem = check_whole_number(samples, 100, "Samples") if opt_mode == "random" else None
        return _input_alert(problem), bool(problem) or not _enough_tickers(tickers, thresh)

    @app.callback(
        Output("optimized-params-store", "data"),
        Output("optimize-result-div", "children"),
        Output("optimize-progress-div", "children", allow_duplicate=True),
        Input("optimize-btn", "n_clicks"),
        State("start-date", "date"),
        State("backtest-start-date", "date"),
        State("backtest-end-date", "date"),
        State("ticker-store", "data"),
        State("thresh-slider", "value"),
        State("select-all-tickers-checkbox", "value"),
        State("top-k-slider", "value"),
        State("weight-method-select", "value"),
        State("tcost-input", "value"),
        State("opt-mode-radio", "value"),
        State("opt-samples-input", "value"),
        State("opt-seed-input", "value"),
        State("strategy-mode-radio", "value"),
        State("core-weight-input", "value"),
        State("core-config-store", "data"),
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
        backtest_end_date,
        selected_tickers,
        thresh,
        select_all,
        top_k,
        weight_method,
        tcost,
        opt_mode,
        n_samples,
        opt_seed,
        strategy_mode,
        core_weight_pct,
        core_config,
    ):
        """Run parameter optimization."""
        if not n_clicks or not selected_tickers or len(selected_tickers) < 2:
            raise PreventUpdate

        # Validate inputs (samples also disable the button; shared sidebar inputs are checked here)
        problems = [check_whole_number(n_samples, 100, "Samples") if opt_mode == "random" else None,
                    check_backtest_dates(backtest_start_date, backtest_end_date), check_tcost(tcost)]
        try:
            problem = next((p for p in problems if p), None)
            if problem:
                raise AnalysisError(problem)
            opt_core_config = (parse_core_config(core_config, core_weight_pct)
                               if strategy_mode == "core-satellite" else None)
            opt_seed = parse_seed(opt_seed) if opt_mode == "random" else None
        except AnalysisError as err:
            return None, _error_alert(err), ""
        n_samples = int(n_samples) if opt_mode == "random" else None

        # Load data
        dataset, _ = load_price_data(selected_tickers, start_date=start_date)
        universe = list(dataset.columns)  # selectable tickers; core-only tickers are appended later

        # Handle parameters
        if select_all:
            top_k = None
        tcost = float(tcost) / 100.0

        # Core-only tickers get prices too (they never enter the signal: see universe=)
        if opt_core_config:
            dataset, _ = with_core_prices(dataset, opt_core_config["tickers"], start_date)

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
            end_date=backtest_end_date,
            thresh=thresh,
            n_samples=n_samples,
            seed=opt_seed,
            progress_callback=update_progress,
            core_config=opt_core_config,
            universe=universe,
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

    @app.callback(
        Output("wf-train-warning", "children"),
        Output("wf-test-warning", "children"),
        Output("wf-samples-warning", "children"),
        Output("walk-forward-btn", "disabled"),
        Input("wf-train-input", "value"),
        Input("wf-test-input", "value"),
        Input("wf-samples-input", "value"),
        Input("ticker-store", "data"),
        Input("thresh-slider", "value"),
    )
    def validate_wf_inputs(train_months, test_months, samples, tickers, thresh):
        """Only writer of walk-forward-btn.disabled: train / test / samples and ticker count."""
        problems = _wf_input_problems(train_months, test_months, samples)
        disabled = any(problems) or not _enough_tickers(tickers, thresh)
        return (*(_input_alert(p) for p in problems), disabled)

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
        State("wf-samples-input", "value"),
        State("wf-seed-input", "value"),
        State("bm-type-radio", "value"),
        State("bm-config-store", "data"),
        State("strategy-mode-radio", "value"),
        State("core-weight-input", "value"),
        State("core-config-store", "data"),
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
        wf_samples,
        wf_seed,
        bm_type,
        bm_config,
        strategy_mode,
        core_weight_pct,
        core_config,
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

        # Validate inputs (the button is disabled for these already; this guards stale states)
        problems = [p for p in (*_wf_input_problems(train_months, test_months, wf_samples), check_tcost(tcost)) if p]
        try:
            if problems:
                raise AnalysisError(problems[0])
            wf_core_config = (parse_core_config(core_config, core_weight_pct)
                              if strategy_mode == "core-satellite" else None)
            wf_seed = parse_seed(wf_seed)
        except AnalysisError as err:
            return (None, _error_alert(err), "") + empty_results
        train_months, test_months, wf_samples = int(train_months), int(test_months), int(wf_samples)

        # Load data
        dataset, _ = load_price_data(selected_tickers, start_date=start_date)
        universe = list(dataset.columns)  # selectable tickers; core-only tickers are appended later

        # Load benchmark data if custom
        bm_data = None
        wf_bm_label = None
        if bm_type == "custom" and bm_config:
            bm_series, bm_label, _ = build_composite_bm(bm_config, start_date)
            if bm_series is not None:
                bm_data = bm_series
                wf_bm_label = bm_label

        # Handle parameters
        if select_all:
            top_k = None
        tcost = float(tcost) / 100.0

        # Core-only tickers get prices too (they never enter the signal: see universe=)
        if wf_core_config:
            dataset, _ = with_core_prices(dataset, wf_core_config["tickers"], start_date)

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
            train_months=train_months,
            test_months=test_months,
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
            core_config=wf_core_config,
            universe=universe,
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
                html.Strong("Walk-Forward Complete!"),
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
        oos_nav_info = (f"One continuous backtest over all out-of-sample test periods, each fold trading its own "
                        f"parameters (switching folds is a normal rebalance). BM = {bm_name}")

        fig_oos = go.Figure()
        sharpe_comparison = ""

        if wf_store["combined_oos_nav"]:
            oos_nav = pd.read_json(StringIO(wf_store["combined_oos_nav"]), typ="series")
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
                oos_bm_nav = pd.read_json(StringIO(wf_store["combined_oos_bm_nav"]), typ="series")
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
            oos_bm_sharpe = sharpe_from_nav(oos_bm_nav)
            if not np.isnan(oos_bm_sharpe):
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
