"""Sidebar callbacks: collapse toggles, ticker selection, core / benchmark rows."""

import dash
from dash import Input, Output, State, html, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from src.analysis import check_tcost
from src.config import TICKER_PRESETS, DEFAULT_PRESET, DEFAULT_CORE_WEIGHT_PCT
from dash_components.sidebar_layout import _create_bm_row, _create_core_row


def register(app):
    """Register sidebar callbacks."""

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
        Output("core-satellite-div", "style"),
        Output("satellite-weight-label", "children"),
        Input("strategy-mode-radio", "value"),
        Input("core-weight-input", "value"),
    )
    def toggle_core_satellite(mode, core_pct):
        """Show/hide core-satellite settings and update satellite label."""
        if mode == "core-satellite":
            if core_pct is None or not 1 <= core_pct <= 99:
                return {"display": "block"}, html.Span("Core allocation must be 1–99%", className="text-danger")
            return {"display": "block"}, f"Satellite: {100 - core_pct:g}%"
        return {"display": "none"}, f"Satellite: {100 - DEFAULT_CORE_WEIGHT_PCT}%"

    @app.callback(
        Output("tcost-warning", "children"),
        Input("tcost-input", "value"),
    )
    def validate_tcost(tcost):
        """Inline hint; Run / Optimize / Walk-Forward refuse a blank or invalid cost when clicked."""
        return check_tcost(tcost) or ""

    @app.callback(
        Output("core-rows-container", "children"),
        Input("core-add-row-btn", "n_clicks"),
        Input({"type": "core-row-delete", "index": ALL}, "n_clicks"),
        State("core-rows-container", "children"),
        prevent_initial_call=True,
    )
    def manage_core_rows(add_clicks, delete_clicks, current_rows):
        """Add or delete core ticker rows."""
        from dash import ctx

        triggered = ctx.triggered_id

        if not current_rows:
            current_rows = []

        if triggered == "core-add-row-btn":
            existing_indices = []
            for row in current_rows:
                if isinstance(row, dict) and "props" in row:
                    row_id = row["props"].get("id", {})
                    if isinstance(row_id, dict):
                        existing_indices.append(row_id.get("index", 0))
            next_index = max(existing_indices, default=-1) + 1
            current_rows.append(_create_core_row(next_index))
            return current_rows

        if isinstance(triggered, dict) and triggered.get("type") == "core-row-delete":
            delete_index = triggered["index"]
            for i, row in enumerate(current_rows):
                if isinstance(row, dict) and "props" in row:
                    row_id = row["props"].get("id", {})
                    if isinstance(row_id, dict) and row_id.get("index") == delete_index:
                        if delete_clicks and i < len(delete_clicks) and delete_clicks[i]:
                            new_rows = [r for j, r in enumerate(current_rows) if j != i]
                            return new_rows if new_rows else [_create_core_row(0)]
                        break
            raise PreventUpdate

        raise PreventUpdate

    @app.callback(
        Output("core-config-store", "data"),
        Output("core-ticker-status", "children"),
        Input("apply-core-btn", "n_clicks"),
        State({"type": "core-row-ticker", "index": ALL}, "value"),
        State({"type": "core-row-weight", "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def apply_core_config(n_clicks, tickers, weights):
        """Apply core config from all rows into core-config-store."""
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

        # Display normalized weights
        total = sum(c["weight"] for c in config)
        label = " + ".join(f"{c['ticker']} {c['weight']/total*100:.1f}%" for c in config)
        return config, dbc.Alert(
            [html.I(className="fas fa-check-circle me-2"), f"Core: {label}"],
            color="success",
            className="py-2 mb-0",
        )

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

        label = " + ".join(f"{c['ticker']} {c['weight']:.1f}%" for c in config)
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

    @app.callback(
        Output("ticker-thresh-warning", "children"),
        Output("run-analysis-btn", "disabled"),
        Input("ticker-store", "data"),
        Input("thresh-slider", "value"),
        Input("run-analysis-busy", "data"),
    )
    def validate_run_button(selected_tickers, thresh, busy):
        """Only writer of run-analysis-btn.disabled: ticker count, and off while a run is in progress.
        Other inputs are checked by run_analysis() when clicked."""
        tickers = selected_tickers or []
        n = len(tickers)
        if n < thresh:
            warning = dbc.Alert(
                [
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"Selected tickers ({n}) < Min Valid Tickers ({thresh}). "
                    "Please add more tickers or lower the threshold.",
                ],
                color="warning",
                className="py-2 mb-0",
            )
            return warning, True  # Optimize / Walk-forward buttons check the ticker count themselves
        return None, bool(busy)
