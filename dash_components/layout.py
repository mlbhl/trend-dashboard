"""Layout definition for Trend Dashboard Dash app."""

from datetime import date
import dash_bootstrap_components as dbc
from dash import dcc, html

from src.config import (
    ALPHA_LIST,
    DEFAULT_START_DATE,
    DEFAULT_BACKTEST_START_DATE,
    DEFAULT_SHORT_WINDOW,
    DEFAULT_MID_WINDOW,
    DEFAULT_LONG_WINDOW,
    DEFAULT_SHORT_WEIGHT,
    DEFAULT_MID_WEIGHT,
    DEFAULT_LONG_WEIGHT,
    DEFAULT_TOP_K,
    DEFAULT_N_QUANTILES,
    DEFAULT_THRESH,
    WEIGHT_METHODS,
)


def label_with_help(label_text: str, tooltip_id: str, tooltip_text: str, size: str = None):
    """Create a label with a help icon (?) that shows a tooltip on hover."""
    label_style = {"marginRight": "5px"}
    if size == "sm":
        label_style["fontSize"] = "0.875rem"

    return html.Div(
        [
            html.Span(label_text, style=label_style),
            html.Span(
                html.I(className="fas fa-question-circle text-muted"),
                id=tooltip_id,
                style={"cursor": "pointer", "fontSize": "0.8rem"},
            ),
            dbc.Tooltip(
                tooltip_text,
                target=tooltip_id,
                placement="right",
            ),
        ],
        className="d-flex align-items-center mb-1",
    )


def create_sidebar_content():
    """Create the sidebar content (controls only, no wrapper)."""
    return [
        html.H4("Parameters", className="mb-3"),

            # Data Settings Section
            html.H6("Data Settings", className="text-muted mb-2"),
            label_with_help("Download Start Date", "help-start-date", "Data download period start"),
            dcc.DatePickerSingle(
                id="start-date",
                date=DEFAULT_START_DATE,
                min_date_allowed="1990-01-01",
                max_date_allowed=date.today(),
                display_format="YYYY-MM-DD",
                className="mb-2 w-100",
            ),
            label_with_help("Backtest Start Date", "help-backtest-date", "If data is insufficient, the earliest available date will be used."),
            dcc.DatePickerSingle(
                id="backtest-start-date",
                date=DEFAULT_BACKTEST_START_DATE,
                min_date_allowed="1990-01-01",
                max_date_allowed=date.today(),
                display_format="YYYY-MM-DD",
                className="mb-3 w-100",
            ),
            html.Hr(),

            # Ticker Selection Section
            html.H6("Ticker Selection", className="text-muted mb-2"),
            label_with_help("Tickers", "help-tickers", "ETFs to include in the analysis. Click X to remove."),
            dcc.Store(id="ticker-store", data=ALPHA_LIST.copy()),
            html.Div(id="ticker-tags", className="mb-2", style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
            label_with_help("Add Ticker", "help-add-ticker", "Enter ticker symbol and press Enter to add"),
            dbc.InputGroup(
                [
                    dbc.Input(
                        id="new-ticker-input",
                        placeholder="e.g., SPY",
                        type="text",
                    ),
                    dbc.Button(
                        html.I(className="fas fa-plus"),
                        id="add-ticker-btn",
                        color="secondary",
                        outline=True,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Button(
                "Reset to Default",
                id="reset-tickers-btn",
                color="link",
                size="sm",
                className="mb-2 p-0",
            ),
            label_with_help("Min Valid Tickers", "help-thresh", "Minimum number of valid tickers required to run the analysis"),
            dcc.Slider(
                id="thresh-slider",
                min=2,
                max=20,
                step=1,
                value=DEFAULT_THRESH,
                marks={i: str(i) for i in [2, 5, 10, 15, 20]},
                tooltip={"placement": "bottom", "always_visible": False},
                className="mb-3",
            ),
            html.Hr(),

            # Signal Window Parameters Section
            html.Div(
                [
                    html.H6("Signal Window Parameters", className="text-muted d-inline me-2"),
                    html.Span(
                        html.I(className="fas fa-question-circle text-muted"),
                        id="help-signal-windows",
                        style={"cursor": "pointer", "fontSize": "0.8rem"},
                    ),
                    dbc.Tooltip(
                        "Lookback periods for momentum calculation. Shorter = more reactive, Longer = more stable.",
                        target="help-signal-windows",
                        placement="right",
                    ),
                ],
                className="mb-2",
            ),
            dbc.Label("Short Window (months)"),
            dcc.Slider(
                id="short-window-slider",
                min=1,
                max=12,
                step=1,
                value=DEFAULT_SHORT_WINDOW,
                marks={i: str(i) for i in [1, 3, 6, 9, 12]},
                tooltip={"placement": "bottom", "always_visible": False},
                className="mb-2",
            ),
            dbc.Label("Mid Window (months)"),
            dcc.Slider(
                id="mid-window-slider",
                min=1,
                max=12,
                step=1,
                value=DEFAULT_MID_WINDOW,
                marks={i: str(i) for i in [1, 3, 6, 9, 12]},
                tooltip={"placement": "bottom", "always_visible": False},
                className="mb-2",
            ),
            dbc.Label("Long Window (months)"),
            dcc.Slider(
                id="long-window-slider",
                min=1,
                max=12,
                step=1,
                value=DEFAULT_LONG_WINDOW,
                marks={i: str(i) for i in [1, 3, 6, 9, 12]},
                tooltip={"placement": "bottom", "always_visible": False},
                className="mb-3",
            ),
            html.Hr(),

            # Signal Weight Parameters Section
            html.Div(
                [
                    html.H6("Signal Weight Parameters", className="text-muted d-inline me-2"),
                    html.Span(
                        html.I(className="fas fa-question-circle text-muted"),
                        id="help-signal-weights",
                        style={"cursor": "pointer", "fontSize": "0.8rem"},
                    ),
                    dbc.Tooltip(
                        "Relative importance of each window. Final signal = weighted average of 3 momentum ranks.",
                        target="help-signal-weights",
                        placement="right",
                    ),
                ],
                className="mb-2",
            ),
            dbc.RadioItems(
                id="weight-mode-radio",
                options=[
                    {"label": "Equal (1/3 each)", "value": "equal"},
                    {"label": "Custom", "value": "custom"},
                ],
                value="custom",
                inline=True,
                className="mb-2",
            ),
            html.Div(
                id="custom-weight-div",
                children=[
                    dbc.Label("Short Weight"),
                    dcc.Slider(
                        id="short-weight-slider",
                        min=0,
                        max=1,
                        step=0.01,
                        value=DEFAULT_SHORT_WEIGHT,
                        marks={0: "0", 0.5: "0.5", 1: "1"},
                        tooltip={"placement": "bottom", "always_visible": False},
                        className="mb-2",
                    ),
                    dbc.Label("Mid Weight"),
                    dcc.Slider(
                        id="mid-weight-slider",
                        min=0,
                        max=1,
                        step=0.01,
                        value=DEFAULT_MID_WEIGHT,
                        marks={0: "0", 0.5: "0.5", 1: "1"},
                        tooltip={"placement": "bottom", "always_visible": False},
                        className="mb-2",
                    ),
                    dbc.Label("Long Weight"),
                    dcc.Slider(
                        id="long-weight-slider",
                        min=0,
                        max=1,
                        step=0.01,
                        value=DEFAULT_LONG_WEIGHT,
                        marks={0: "0", 0.5: "0.5", 1: "1"},
                        tooltip={"placement": "bottom", "always_visible": False},
                        className="mb-2",
                    ),
                ],
            ),
            html.Hr(),

            # Portfolio Settings Section
            html.H6("Portfolio Settings", className="text-muted mb-2"),
            label_with_help("Strategy Type", "help-strategy-type", "Top-K: Invest in top K ranked assets. Quantile: Divide into groups to analyze momentum factor."),
            dcc.Dropdown(
                id="strategy-type-select",
                options=[
                    {"label": "Top-K", "value": "Top-K"},
                    {"label": "Quantile", "value": "Quantile"},
                ],
                value="Top-K",
                clearable=False,
                className="mb-2",
            ),
            html.Div(
                [
                    dbc.Checkbox(
                        id="select-all-tickers-checkbox",
                        label="Select All Tickers",
                        value=False,
                        className="d-inline",
                    ),
                    html.Span(
                        html.I(className="fas fa-question-circle text-muted ms-2"),
                        id="help-select-all",
                        style={"cursor": "pointer", "fontSize": "0.8rem"},
                    ),
                    dbc.Tooltip(
                        "Select all available tickers at each rebalancing (dynamic K)",
                        target="help-select-all",
                        placement="right",
                    ),
                ],
                className="mb-2",
            ),
            html.Div(
                id="top-k-div",
                children=[
                    label_with_help("Top K", "help-top-k", "Number of top-ranked assets to include in portfolio. Lower K = more concentrated."),
                    dcc.Slider(
                        id="top-k-slider",
                        min=1,
                        max=10,
                        step=1,
                        value=DEFAULT_TOP_K,
                        marks={i: str(i) for i in [1, 3, 5, 7, 10]},
                        tooltip={"placement": "bottom", "always_visible": False},
                        className="mb-2",
                    ),
                ],
            ),
            label_with_help("Number of Quantiles", "help-n-quantiles", "Number of groups to divide assets into. Q5=Best, Q1=Worst for 5 quantiles."),
            dcc.Slider(
                id="n-quantiles-slider",
                min=2,
                max=10,
                step=1,
                value=DEFAULT_N_QUANTILES,
                marks={i: str(i) for i in [2, 3, 4, 5, 10]},
                tooltip={"placement": "bottom", "always_visible": False},
                className="mb-2",
            ),
            label_with_help("Weighting Method", "help-weight-method", "Equal: 1/N weight | Inverse Vol: weight by 1/volatility | Rank: weight by 1/rank"),
            dcc.Dropdown(
                id="weight-method-select",
                options=[{"label": k, "value": v} for k, v in WEIGHT_METHODS.items()],
                value="equal",
                clearable=False,
                className="mb-2",
            ),
            label_with_help("Transaction Cost (one-way)", "help-tcost", "Cost per trade in %. Applied to both buy and sell. Example: 0.001 = 0.1% per trade."),
            dbc.Input(
                id="tcost-input",
                type="number",
                min=0,
                max=0.01,
                step=0.0005,
                value=0.0,
                className="mb-3",
            ),
            html.Hr(),

            # Benchmark Settings Section
            html.H6("Benchmark Settings", className="text-muted mb-2"),
            html.Div(
                [
                    dbc.RadioItems(
                        id="bm-type-radio",
                        options=[
                            {"label": "Equal Weight", "value": "equal_weight"},
                            {"label": "Custom Ticker", "value": "custom"},
                        ],
                        value="equal_weight",
                        inline=True,
                        className="d-inline-flex",
                    ),
                    html.Span(
                        html.I(className="fas fa-question-circle text-muted ms-2"),
                        id="help-bm-type",
                        style={"cursor": "pointer", "fontSize": "0.8rem"},
                    ),
                    dbc.Tooltip(
                        "EW uses equal weight of all selected tickers",
                        target="help-bm-type",
                        placement="right",
                    ),
                ],
                className="mb-2",
            ),
            html.Div(
                id="custom-bm-div",
                children=[
                    dbc.InputGroup(
                        [
                            dbc.Input(
                                id="bm-ticker-input",
                                placeholder="e.g., SPY",
                                type="text",
                                #value="SPY",
                                debounce=True,  # Triggers on Enter
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-check")],
                                id="apply-bm-ticker-btn",
                                color="success",
                                outline=True,
                            ),
                        ],
                        className="mb-1",
                    ),
                    dbc.FormText("Enter ticker symbol and press Enter to apply", className="mb-2"),
                    html.Div(id="bm-ticker-status", className="mb-2"),
                ],
                style={"display": "none"},
            ),
            html.Hr(),

            # Run Analysis Button
            html.Div(
                id="run-analysis-btn-container",
                children=[
                    dbc.Button(
                        [html.I(className="fas fa-play me-2", id="run-btn-icon"), "Run Analysis"],
                        id="run-analysis-btn",
                        color="primary",
                        className="w-100 mb-3",
                        size="lg",
                    ),
                ],
            ),

            # Optimization Section (Collapsible)
            html.Div(
                [
                    html.H6("Parameter Optimization", className="text-muted d-inline me-2"),
                    html.Span(
                        html.I(className="fas fa-question-circle text-muted"),
                        id="help-optimization",
                        style={"cursor": "pointer", "fontSize": "0.8rem"},
                    ),
                    dbc.Tooltip(
                        "Grid search to find optimal window/weight parameters. Maximizes Sharpe ratio on Backtest Start Date period.",
                        target="help-optimization",
                        placement="right",
                    ),
                ],
                className="mb-2",
            ),
            html.Small(
                "Optimize Top-K strategy parameters using Backtest Start Date",
                className="text-muted d-block mb-2",
            ),
            html.Div(
                [
                    dbc.RadioItems(
                        id="opt-mode-radio",
                        options=[
                            {"label": "Full Grid", "value": "full"},
                            {"label": "Random Grid", "value": "random"},
                        ],
                        value="random",
                        inline=True,
                        className="d-inline-flex",
                    ),
                    html.Span(
                        html.I(className="fas fa-question-circle text-muted ms-2"),
                        id="help-opt-mode",
                        style={"cursor": "pointer", "fontSize": "0.8rem"},
                    ),
                    dbc.Tooltip(
                        "Full: Test all 2,016 combinations. Random: Sample subset for faster results.",
                        target="help-opt-mode",
                        placement="right",
                    ),
                ],
                className="mb-2",
            ),
            html.Div(
                id="random-opt-div",
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    label_with_help("Samples", "help-opt-samples", "Number of random combinations to test (100~2016).", size="sm"),
                                    dbc.Input(
                                        id="opt-samples-input",
                                        type="number",
                                        value=500,
                                        min=1,
                                        max=2016,
                                        size="sm",
                                    ),
                                    html.Div(id="opt-samples-warning"),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    label_with_help("Seed", "help-opt-seed", "Random seed for reproducible results.", size="sm"),
                                    dbc.Input(
                                        id="opt-seed-input",
                                        type="text",
                                        placeholder="Random",
                                        size="sm",
                                    ),
                                ],
                                width=6,
                            ),
                        ],
                        className="mb-2",
                    ),
                ],
            ),
            dbc.Button(
                [html.I(className="fas fa-search me-2"), "Optimize Parameters"],
                id="optimize-btn",
                color="secondary",
                outline=True,
                className="w-100 mb-2",
            ),
            html.Div(id="optimize-progress-div", className="mb-2"),
            html.Div(id="optimize-result-div", className="mb-3"),
            html.Hr(),

            # Walk-Forward Section (Collapsible)
            html.Div(
                [
                    html.H6("Walk-Forward Analysis", className="text-muted d-inline me-2"),
                    html.Span(
                        html.I(className="fas fa-question-circle text-muted"),
                        id="help-walk-forward",
                        style={"cursor": "pointer", "fontSize": "0.8rem"},
                    ),
                    dbc.Tooltip(
                        "Out-of-sample validation. Trains on past data, tests on future. Measures strategy robustness and overfitting risk.",
                        target="help-walk-forward",
                        placement="right",
                    ),
                ],
                className="mb-2",
            ),
            html.Small(
                "Out-of-sample validation using Download Start Date",
                className="text-muted d-block mb-2",
            ),
            dbc.Button(
                "Walk-Forward Settings",
                id="wf-settings-collapse-btn",
                color="link",
                size="sm",
                className="p-0 mb-2",
            ),
            dbc.Collapse(
                id="wf-settings-collapse",
                is_open=False,
                children=[
                    html.Div(
                        [
                            dbc.Label("Window Type", size="sm", className="d-inline me-2"),
                            html.Span(
                                html.I(className="fas fa-question-circle text-muted"),
                                id="help-wf-window-type",
                                style={"cursor": "pointer", "fontSize": "0.7rem"},
                            ),
                            dbc.Tooltip(
                                "Rolling: Fixed lookback. Expanding: Uses all available history.",
                                target="help-wf-window-type",
                                placement="right",
                            ),
                        ],
                        className="mb-1",
                    ),
                    dbc.RadioItems(
                        id="wf-window-type-radio",
                        options=[
                            {"label": "Rolling", "value": "rolling"},
                            {"label": "Expanding", "value": "expanding"},
                        ],
                        value="rolling",
                        inline=True,
                        className="mb-2",
                    ),
                    label_with_help("Train Period (months)", "help-wf-train", "Length of in-sample training period. Longer = more data but less folds.", size="sm"),
                    dbc.Input(
                        id="wf-train-input",
                        type="number",
                        value=36,
                        min=1,
                        step=1,
                        size="sm",
                        className="mb-1",
                    ),
                    html.Div(id="wf-train-warning", className="mb-1"),
                    label_with_help("Test Period (months)", "help-wf-test", "Length of out-of-sample test period for each fold.", size="sm"),
                    dbc.Input(
                        id="wf-test-input",
                        type="number",
                        value=12,
                        min=1,
                        step=1,
                        size="sm",
                        className="mb-2",
                    ),
                    label_with_help("Step Size (months)", "help-wf-step", "How far to advance between folds. Smaller = more overlap.", size="sm"),
                    dbc.Input(
                        id="wf-step-input",
                        type="number",
                        value=12,
                        min=1,
                        step=1,
                        size="sm",
                        className="mb-2",
                    ),
                    label_with_help("Samples per Fold", "help-wf-samples", "Number of random parameter combinations to test per fold (100~5000).", size="sm"),
                    dbc.Input(
                        id="wf-samples-input",
                        type="number",
                        value=500,
                        min=1,
                        max=5000,
                        step=1,
                        size="sm",
                        className="mb-1",
                    ),
                    html.Div(id="wf-samples-warning", className="mb-1"),
                    label_with_help("Seed", "help-wf-seed", "Random seed for reproducible results. Leave empty for random.", size="sm"),
                    dbc.Input(
                        id="wf-seed-input",
                        type="text",
                        placeholder="Random",
                        size="sm",
                        className="mb-2",
                    ),
                ],
            ),
            dbc.Button(
                [html.I(className="fas fa-sync me-2"), "Walk-Forward Optimize"],
                id="walk-forward-btn",
                color="secondary",
                outline=True,
                className="w-100 mb-2",
            ),
            html.Div(id="wf-progress-div", className="mb-2"),
            html.Div(id="wf-result-div", className="mb-3"),
    ]


def create_sidebar():
    """Create the sidebar with responsive behavior."""
    return dbc.Col(
        [
            # Mobile toggle button (shown only on mobile)
            dbc.Button(
                [html.I(className="fas fa-cog me-2"), "Settings"],
                id="mobile-sidebar-toggle",
                color="primary",
                className="d-lg-none w-100 mb-3",
            ),
            # Sidebar content wrapper (collapsible on mobile, always open on desktop via CSS)
            dbc.Collapse(
                id="sidebar-collapse",
                is_open=False,  # Collapsed on mobile by default
                className="sidebar-collapse",
                children=create_sidebar_content(),
            ),
        ],
        lg=3,
        xs=12,
        className="sidebar bg-light p-3",
    )


def create_kpi_card(title, kpi_id):
    """Create a KPI card component."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="card-subtitle mb-2 text-muted"),
                html.Div(id=f"{kpi_id}-cagr", className="mb-1"),
                html.Div(id=f"{kpi_id}-vol", className="mb-1"),
                html.Div(id=f"{kpi_id}-sharpe", className="mb-1"),
                html.Div(id=f"{kpi_id}-mdd", className="mb-1"),
                html.Div(id=f"{kpi_id}-ytd", className="mb-1"),
            ]
        ),
        className="h-100",
    )


def create_main_content():
    """Create the main content area."""
    return dbc.Col(
        [
            html.H2("Momentum Strategy Dashboard", className="mb-4"),

            # Loading indicator (shown while analysis is running)
            html.Div(
                id="analysis-loading",
                style={"display": "none"},
                children=[
                    dbc.Alert(
                        [
                            dbc.Spinner(size="sm", color="primary", spinner_class_name="me-2"),
                            "Running analysis... Please wait.",
                        ],
                        color="warning",
                        className="mb-4",
                    ),
                ],
            ),

            # Warning/Error messages
            html.Div(id="analysis-warnings", className="mb-3"),

            # Initial instructions (shown when no data)
            html.Div(
                id="initial-instructions",
                children=[
                    dbc.Alert(
                        [
                            html.I(className="fas fa-arrow-left me-2"),
                            "Configure parameters in the sidebar and click ",
                            html.Strong("Run Analysis"),
                            " to start.",
                        ],
                        color="info",
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("User Guide", className="card-title"),
                                html.P(
                                    "This dashboard analyzes momentum-based asset rotation strategies across ETFs.",
                                    className="card-text",
                                ),
                                html.Hr(),
                                html.H5("Quick Start"),
                                html.Ol(
                                    [
                                        html.Li("Select tickers in the sidebar (or use defaults)"),
                                        html.Li("Click Run Analysis"),
                                        html.Li("Review the results"),
                                    ]
                                ),
                                html.Hr(),
                                html.H5("Signal Parameters"),
                                html.P(
                                    "Momentum is measured using 3 lookback periods (Short/Mid/Long). "
                                    "For each period, assets are ranked by returns and the final signal "
                                    "is a weighted average of these ranks."
                                ),
                                html.Hr(),
                                html.H5("Portfolio Settings"),
                                html.Ul(
                                    [
                                        html.Li(
                                            [
                                                html.Strong("Top-K: "),
                                                "Invests in top K ranked assets",
                                            ]
                                        ),
                                        html.Li(
                                            [
                                                html.Strong("Quantile: "),
                                                "Divides assets into groups to analyze momentum factor",
                                            ]
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        className="mb-4",
                    ),
                ],
            ),

            # Analysis results (hidden initially)
            html.Div(
                id="analysis-results",
                style={"display": "none"},
                children=[
                    # Loading indicator
                    dcc.Loading(
                        id="loading-analysis",
                        type="default",
                        children=[
                            # KPI Section
                            html.H4("Key Performance Indicators", className="mb-3"),
                            html.Div(id="backtest-period-info", className="text-muted mb-3"),
                            html.Div(id="kpi-cards-container", className="mb-4"),

                            # User Guide Collapse
                            dbc.Button(
                                [html.I(className="fas fa-book me-2"), "User Guide"],
                                id="user-guide-collapse-btn",
                                color="link",
                                className="mb-2 p-0",
                            ),
                            dbc.Collapse(
                                id="user-guide-collapse",
                                is_open=False,
                                children=dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H5("Quick Reference"),
                                            html.H6("Signal Parameters"),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        "Signal Window: Lookback period in months for each timeframe"
                                                    ),
                                                    html.Li(
                                                        "Signal Weight: Weight for each window (Equal 1/3 or Custom)"
                                                    ),
                                                ]
                                            ),
                                            html.H6("Strategy Types"),
                                            html.Ul(
                                                [
                                                    html.Li("Top-K: Invests in top K ranked assets"),
                                                    html.Li(
                                                        "Quantile: Divides assets into groups to analyze momentum factor"
                                                    ),
                                                ]
                                            ),
                                            html.H6("Weighting Methods"),
                                            html.Ul(
                                                [
                                                    html.Li("Equal: 1/N weight for each asset"),
                                                    html.Li(
                                                        "Inverse Vol: Higher weight to lower volatility assets"
                                                    ),
                                                    html.Li("Rank: Higher weight to better ranked assets"),
                                                ]
                                            ),
                                        ]
                                    ),
                                    className="mb-4",
                                ),
                            ),

                            # Performance Charts Section
                            html.H4("Performance Charts", className="mb-3"),
                            dcc.Graph(id="nav-chart", config={"displaylogo": False}),
                            dcc.Graph(id="drawdown-chart", config={"displaylogo": False}),

                            # Signal Category Table
                            html.H4("Current Signal by Quantile", className="mb-3"),
                            html.Div(id="signal-date-info", className="text-muted mb-2"),
                            html.Small(
                                "Rank 1 = Best (Q5), Higher Rank = Worse (Q1). Format: Ticker (Rank)",
                                className="text-muted d-block mb-2",
                            ),
                            dcc.Graph(id="signal-table-chart", config={"displaylogo": False}),

                            # Raw Signal Data Collapse
                            dbc.Button(
                                "Show Raw Signal Data",
                                id="raw-signal-collapse-btn",
                                color="link",
                                className="mb-2 p-0",
                            ),
                            dbc.Collapse(
                                id="raw-signal-collapse",
                                is_open=False,
                                children=[
                                    html.H5("Latest Signal Ranking", className="mt-2"),
                                    html.Div(id="raw-signal-table-container"),
                                ],
                                className="mb-4",
                            ),

                            # Quantile Spread (only for Quantile strategy)
                            html.Div(
                                id="quantile-spread-section",
                                children=[
                                    html.H4("Quantile Spread (Q5 - Q1)", className="mb-3"),
                                    dcc.Graph(id="spread-chart", config={"displaylogo": False}),
                                ],
                            ),

                            # Holdings Heatmap Section
                            html.H4("Holdings Analysis", className="mb-3"),
                            html.Small(
                                "Q5 = Best performers (Rank 1~), Q1 = Worst performers",
                                className="text-muted d-block mb-2",
                            ),
                            html.Div(
                                id="quantile-selector-div",
                                children=[
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Select Quantile"),
                                                    dcc.Dropdown(
                                                        id="heatmap-quantile-select",
                                                        options=[],
                                                        value=None,
                                                        clearable=False,
                                                    ),
                                                ],
                                                width=3,
                                            ),
                                            dbc.Col(
                                                html.Div(id="quantile-info-div"),
                                                width=9,
                                                className="d-flex align-items-end",
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                ],
                            ),
                            dcc.Graph(id="heatmap-chart", config={"displaylogo": False}),

                            # Performance Statistics Table
                            html.H4("Performance Statistics", className="mb-3"),
                            dcc.Graph(id="stats-table-chart", config={"displaylogo": False}),

                            # Stats DataFrame Collapse
                            dbc.Button(
                                "Show as DataFrame",
                                id="stats-df-collapse-btn",
                                color="link",
                                className="mb-2 p-0",
                            ),
                            dbc.Collapse(
                                id="stats-df-collapse",
                                is_open=False,
                                children=html.Div(id="stats-df-container"),
                                className="mb-4",
                            ),

                            # Walk-Forward Results Section
                            html.Div(
                                id="walk-forward-results-section",
                                style={"display": "none"},
                                children=[
                                    html.Hr(),
                                    html.H4(id="wf-results-title", className="mb-3"),

                                    # WF Summary Metrics
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        [
                                                            html.H6(
                                                                "In-Sample Sharpe (avg)",
                                                                className="text-muted",
                                                            ),
                                                            html.H4(id="wf-is-sharpe"),
                                                        ]
                                                    )
                                                ),
                                                width=3,
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        [
                                                            html.H6(
                                                                "Out-of-Sample Sharpe (avg)",
                                                                className="text-muted",
                                                            ),
                                                            html.H4(id="wf-oos-sharpe"),
                                                        ]
                                                    )
                                                ),
                                                width=3,
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        [
                                                            html.H6(
                                                                "Combined OOS Sharpe",
                                                                className="text-muted",
                                                            ),
                                                            html.H4(id="wf-combined-sharpe"),
                                                        ]
                                                    )
                                                ),
                                                width=3,
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        [
                                                            html.H6("Sharpe Decay", className="text-muted"),
                                                            html.H4(id="wf-sharpe-decay"),
                                                        ]
                                                    )
                                                ),
                                                width=3,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),

                                    # WF Interpretation
                                    html.Div(id="wf-interpretation-div", className="mb-3"),

                                    # Fold Results Table
                                    html.H5("Fold Results", className="mb-3"),
                                    html.Div(id="wf-fold-table-container", className="mb-4"),

                                    # Recommended Parameters
                                    html.Div(
                                        id="wf-recommended-params-section",
                                        children=[
                                            html.H5("Recommended Parameters", className="mb-3"),
                                            html.Div(id="wf-train-period-info", className="text-muted mb-2"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dbc.Card(
                                                            dbc.CardBody(
                                                                [
                                                                    html.H6(
                                                                        "Windows (S/M/L)",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H5(id="wf-rec-windows"),
                                                                ]
                                                            )
                                                        ),
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Card(
                                                            dbc.CardBody(
                                                                [
                                                                    html.H6(
                                                                        "Weights (S/M/L)",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H5(id="wf-rec-weights"),
                                                                ]
                                                            )
                                                        ),
                                                        width=4,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Card(
                                                            dbc.CardBody(
                                                                [
                                                                    html.H6(
                                                                        "In-Sample Sharpe",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H5(id="wf-rec-sharpe"),
                                                                ]
                                                            )
                                                        ),
                                                        width=4,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            dbc.Button(
                                                [
                                                    html.I(className="fas fa-check me-2"),
                                                    "Apply Recommended Params",
                                                ],
                                                id="apply-wf-params-btn",
                                                color="success",
                                                className="mb-4",
                                            ),
                                        ],
                                    ),

                                    # Parameter Stability
                                    html.Div(
                                        id="wf-stability-section",
                                        children=[
                                            html.H5("Parameter Stability", className="mb-3"),
                                            html.Small(
                                                "How much do optimal parameters vary across folds? Lower std = more stable.",
                                                className="text-muted d-block mb-2",
                                            ),
                                            html.Div(id="wf-stability-table-container", className="mb-4"),
                                        ],
                                    ),

                                    # Combined OOS NAV Chart
                                    html.H5("Combined Out-of-Sample NAV", className="mb-3"),
                                    html.Div(id="wf-oos-nav-info", className="text-muted mb-2"),
                                    dcc.Graph(id="wf-oos-nav-chart", config={"displaylogo": False}),
                                    html.Div(id="wf-oos-sharpe-comparison", className="text-muted mb-3"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
        lg=9,
        xs=12,
        className="main-content p-3 p-lg-4",
    )


def create_layout():
    """Create the complete app layout."""
    return dbc.Container(
        [
            # Stores for data
            dcc.Store(id="dataset-store"),
            dcc.Store(id="bm-data-store"),
            dcc.Store(id="params-store"),
            dcc.Store(id="nav-store"),
            dcc.Store(id="signal-store"),
            dcc.Store(id="weights-store"),
            dcc.Store(id="walk-forward-store"),
            dcc.Store(id="optimized-params-store"),

            # Main layout
            dbc.Row(
                [create_sidebar(), create_main_content()],
                className="g-0",
            ),
        ],
        fluid=True,
        className="p-0",
    )
