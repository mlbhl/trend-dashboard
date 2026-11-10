"""Results callbacks: Run Analysis, charts, heatmaps, collapsible sections and Excel downloads."""

from io import StringIO

import pandas as pd
from dash import Input, Output, State, dcc, html, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from src.analysis import AnalysisError, run_analysis
from src.config import TICKER_DESCRIPTIONS
from src.signals import get_signal_ranking, signal_history
from src.metrics import (
    summary_stats,
    rebase,
    calculate_kpis,
    annual_returns,
    monthly_returns,
    format_stats,
    kpi_lines,
)
from src.charts import (
    create_nav_chart,
    create_drawdown_chart,
    create_holding_heatmap,
    create_quantile_holding_heatmap,
    create_signal_category_table,
    create_quantile_spread_chart,
    create_returns_table,
    create_annual_returns_chart,
    create_monthly_returns_table,
)


def _read_frame(data: str) -> pd.DataFrame:
    """DataFrame back from dcc.Store JSON, with a sorted DatetimeIndex."""
    frame = pd.read_json(StringIO(data))
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def _holdings_heatmap(weights_data: dict, params: dict, signal: pd.DataFrame | None, use_name: bool) -> go.Figure:
    """Holdings over the last 24 rebalances: all quantiles colour-coded, or the single Top-K portfolio."""
    if params["strategy_type"] == "Quantile":
        q_wgts = {q: _read_frame(w).iloc[-24:] for q, w in weights_data.items()}
        n_tickers = len(set().union(*(w.columns for w in q_wgts.values())))
        return create_quantile_holding_heatmap(
            q_wgts,
            title="Monthly Holdings by Quantile",
            height=max(500, n_tickers * 25),
            ticker_descriptions=TICKER_DESCRIPTIONS,
            use_name=use_name,
            signal=signal,
        )
    recent = _read_frame(next(iter(weights_data.values()))).iloc[-24:]  # Top-K: one portfolio
    return create_holding_heatmap(
        recent,
        title="Monthly Holdings",
        height=max(500, len(recent.columns) * 25),
        ticker_descriptions=TICKER_DESCRIPTIONS,
        use_name=use_name,
    )


def register(app):
    """Register results callbacks."""

    @app.callback(
        Output("params-store", "data"),
        Output("nav-store", "data"),
        Output("signal-store", "data"),
        Output("weights-store", "data"),
        Output("initial-instructions", "style"),
        Output("analysis-results", "style"),
        Output("analysis-loading", "style"),
        Output("analysis-warnings", "children"),
        Input("run-analysis-btn", "n_clicks"),
        State("start-date", "date"),
        State("backtest-start-date", "date"),
        State("backtest-end-date", "date"),
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
        State("strategy-mode-radio", "value"),
        State("core-weight-input", "value"),
        State("core-config-store", "data"),
        prevent_initial_call=True,
        background=True,
        running=[
            # the button's disabled has one writer (validate_run_button), which reads this flag
            (Output("run-analysis-busy", "data"), True, False),
            (Output("run-analysis-btn", "children"), [dbc.Spinner(size="sm", spinner_class_name="me-2"), "Running..."], [html.I(className="fas fa-play me-2"), "Run Analysis"]),
            (Output("analysis-loading", "style"), {"display": "block"}, {"display": "none"}),
        ],
    )
    def run_analysis_callback(
        n_clicks,
        start_date,
        backtest_start_date,
        backtest_end_date,
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
        strategy_mode,
        core_weight_pct,
        core_config,
    ):
        """Run the main analysis (computation lives in src/analysis.py)."""
        if not n_clicks or not selected_tickers or len(selected_tickers) < 2:
            raise PreventUpdate

        try:
            result = run_analysis(
                tickers=selected_tickers,
                start_date=start_date,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
                thresh=thresh,
                short_window=short_window,
                mid_window=mid_window,
                long_window=long_window,
                weight_mode=weight_mode,
                short_wgt=short_wgt,
                mid_wgt=mid_wgt,
                long_wgt=long_wgt,
                strategy_type=strategy_type,
                select_all=select_all,
                top_k=top_k,
                n_quantiles=n_quantiles,
                weight_method=weight_method,
                tcost_pct=tcost,
                bm_type=bm_type,
                bm_config=bm_config,
                strategy_mode=strategy_mode,
                core_weight_pct=core_weight_pct,
                core_config=core_config,
            )
        except AnalysisError as err:
            return (
                None, None, None, None,
                {"display": "block"},  # Show instructions
                {"display": "none"},  # Hide results
                {"display": "none"},  # Hide loading
                dbc.Alert([html.I(className="fas fa-times-circle me-2"), str(err)], color="danger"),
            )

        warnings = [
            dbc.Alert(
                [html.I(className="fas fa-exclamation-triangle me-2"), msg],
                color="warning",
                dismissable=True,
            )
            for msg in result.warnings
        ]

        return (
            result.params,
            result.navs.to_json(date_format="iso"),
            result.signal.to_json(date_format="iso"),
            {name: w.to_json(date_format="iso") for name, w in result.weights.items()},
            {"display": "none"},  # Hide instructions
            {"display": "block"},  # Show results
            {"display": "none"},  # Hide loading
            warnings if warnings else "",  # Show warnings
        )

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
        Output("annual-returns-chart", "figure"),
        Output("monthly-returns-select", "options"),
        Output("monthly-returns-select", "value"),
        Output("heatmap-chart", "figure"),
        Output("stats-table-chart", "figure"),
        Output("stats-df-container", "children"),
        Input("nav-store", "data"),
        State("params-store", "data"),
        State("signal-store", "data"),
        State("weights-store", "data"),
        State("heatmap-label-toggle", "value"),
    )
    def update_charts(navs_json, params, signal_json, weights_data, label_mode):
        """Update all charts when NAV data changes."""
        if not navs_json or not params:
            raise PreventUpdate

        # Parse data
        navs = _read_frame(navs_json)

        signal = _read_frame(signal_json)

        # Rebase NAVs
        navs_rebased = rebase(navs)

        # Backtest period info
        period_info = f"Backtest Period: {params['display_start']} ~ {params['display_end']}"

        # KPI Cards
        bm_col = params.get("bm_col_name", "BM (Equal Weight)")
        kpis = calculate_kpis(navs, benchmark_col=bm_col)
        kpi_cards = []
        for name, metrics in kpis.items():
            lines = kpi_lines(metrics)
            card = dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [html.H5(name, className="card-title")]
                        + [html.P(line, className="mb-1") for line in lines[:-1]]
                        + [html.P(lines[-1], className="mb-0")]
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

        # Signal History — 마지막 달이 미완이면 컬럼명을 실제 데이터 일자로 표기
        signal_display = signal_history(signal, params["data_end"])
        raw_signal_table = dash_table.DataTable(
            data=signal_display.reset_index().to_dict("records"),
            columns=[{"name": col, "id": col} for col in ["index"] + list(signal_display.columns)],
            style_cell={"textAlign": "center"},
            style_header={"fontWeight": "bold"},
        )

        # Annual Returns Chart
        ann_rets = annual_returns(navs)
        fig_annual = create_annual_returns_chart(ann_rets, title="Annual Returns")

        # Monthly Returns Dropdown options (all NAV columns)
        monthly_options = [{"label": col, "value": col} for col in navs.columns]
        monthly_default = navs.columns[0]

        # Quantile Spread (only for Quantile strategy)
        n_q = params.get("n_quantiles", 5)
        top_q = f"Q{n_q}"
        if params["strategy_type"] == "Quantile" and "Q1" in navs.columns and top_q in navs.columns:
            spread_style = {"display": "block"}
            fig_spread = create_quantile_spread_chart(navs, top_q=top_q)
        else:
            spread_style = {"display": "none"}
            fig_spread = go.Figure()

        # Heatmap
        fig_heatmap = (_holdings_heatmap(weights_data, params, signal, label_mode == "name")
                       if weights_data else go.Figure())

        # Stats Table
        stats = summary_stats(navs)
        fig_stats = create_returns_table(stats)

        # Stats DataFrame (formatted like the table)
        formatted_stats = format_stats(stats)
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
            fig_annual,
            monthly_options,
            monthly_default,
            fig_heatmap,
            fig_stats,
            stats_df_table,
        )

    @app.callback(
        Output("heatmap-chart", "figure", allow_duplicate=True),
        Input("heatmap-label-toggle", "value"),
        State("weights-store", "data"),
        State("params-store", "data"),
        State("signal-store", "data"),
        prevent_initial_call=True,
    )
    def update_heatmap_label(label_mode, weights_data, params, signal_json):
        """Toggle y-axis labels between ticker and name."""
        if not weights_data or not params:
            raise PreventUpdate
        signal = _read_frame(signal_json) if signal_json else None
        return _holdings_heatmap(weights_data, params, signal, label_mode == "name")

    @app.callback(
        Output("monthly-returns-chart", "figure"),
        Input("monthly-returns-select", "value"),
        State("nav-store", "data"),
        prevent_initial_call=True,
    )
    def update_monthly_returns(selected_col, navs_json):
        """Update monthly returns heatmap when dropdown selection changes."""
        if not selected_col or not navs_json:
            raise PreventUpdate

        navs = _read_frame(navs_json)

        if selected_col not in navs.columns:
            raise PreventUpdate

        m_rets = monthly_returns(navs[selected_col])
        fig = create_monthly_returns_table(m_rets, title=f"Monthly Returns — {selected_col}")
        return fig

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

    @app.callback(
        Output("download-nav", "data"),
        Input("dl-nav-btn", "n_clicks"),
        State("nav-store", "data"),
        prevent_initial_call=True,
    )
    def download_nav(n_clicks, navs_json):
        """Download rebased NAV data as Excel."""
        if not n_clicks or not navs_json:
            raise PreventUpdate
        from io import BytesIO
        navs = _read_frame(navs_json)
        navs_rebased = rebase(navs)
        buf = BytesIO()
        navs_rebased.to_excel(buf, index=True, sheet_name="NAV")
        buf.seek(0)
        return dcc.send_bytes(buf.getvalue(), "nav_data.xlsx")

    @app.callback(
        Output("download-dd", "data"),
        Input("dl-dd-btn", "n_clicks"),
        State("nav-store", "data"),
        prevent_initial_call=True,
    )
    def download_drawdown(n_clicks, navs_json):
        """Download drawdown data as Excel."""
        if not n_clicks or not navs_json:
            raise PreventUpdate
        from io import BytesIO
        navs = _read_frame(navs_json)
        navs_rebased = rebase(navs)
        cummax = navs_rebased.cummax()
        drawdown = (navs_rebased - cummax) / cummax
        buf = BytesIO()
        drawdown.to_excel(buf, index=True, sheet_name="Drawdown")
        buf.seek(0)
        return dcc.send_bytes(buf.getvalue(), "drawdown_data.xlsx")

    @app.callback(
        Output("download-annual", "data"),
        Input("dl-annual-btn", "n_clicks"),
        State("nav-store", "data"),
        prevent_initial_call=True,
    )
    def download_annual_returns(n_clicks, navs_json):
        """Download annual returns data as Excel."""
        if not n_clicks or not navs_json:
            raise PreventUpdate
        from io import BytesIO
        navs = _read_frame(navs_json)
        ann_rets = annual_returns(navs)
        buf = BytesIO()
        ann_rets.to_excel(buf, index=True, sheet_name="Annual Returns")
        buf.seek(0)
        return dcc.send_bytes(buf.getvalue(), "annual_returns.xlsx")

    @app.callback(
        Output("download-monthly", "data"),
        Input("dl-monthly-btn", "n_clicks"),
        State("nav-store", "data"),
        State("monthly-returns-select", "value"),
        prevent_initial_call=True,
    )
    def download_monthly_returns(n_clicks, navs_json, selected_col):
        """Download monthly returns data as Excel."""
        if not n_clicks or not navs_json or not selected_col:
            raise PreventUpdate
        from io import BytesIO
        navs = _read_frame(navs_json)
        if selected_col not in navs.columns:
            raise PreventUpdate
        m_rets = monthly_returns(navs[selected_col])
        m_rets.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        buf = BytesIO()
        m_rets.to_excel(buf, index=True, sheet_name="Monthly Returns")
        buf.seek(0)
        return dcc.send_bytes(buf.getvalue(), "monthly_returns.xlsx")

    @app.callback(
        Output("download-holdings", "data"),
        Input("dl-holdings-btn", "n_clicks"),
        State("weights-store", "data"),
        State("params-store", "data"),
        prevent_initial_call=True,
    )
    def download_holdings(n_clicks, weights_data, params):
        """Download holdings/weights data as Excel."""
        if not n_clicks or not weights_data:
            raise PreventUpdate
        from io import BytesIO
        buf = BytesIO()
        if params and params.get("strategy_type") == "Quantile":
            with pd.ExcelWriter(buf) as writer:
                for q_name in sorted(weights_data.keys(), key=lambda x: int(x[1:])):
                    wgt = _read_frame(weights_data[q_name])
                    wgt.to_excel(writer, index=True, sheet_name=q_name)
        else:
            wgt = _read_frame(next(iter(weights_data.values())))  # Top-K stores exactly one portfolio
            wgt.to_excel(buf, index=True, sheet_name="Holdings")
        buf.seek(0)
        return dcc.send_bytes(buf.getvalue(), "holdings_data.xlsx")
