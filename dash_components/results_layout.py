"""Results area layout: user guide texts and one function per results section."""

import dash_bootstrap_components as dbc
from dash import html
from dash import dcc


def create_instructions():
    """Initial instructions and user guide, shown until the first analysis."""
    return html.Div(
        id="initial-instructions",
        children=[
            dbc.Alert(
                [
                    html.I(className="fas fa-arrow-left me-2"),
                    "Configure parameters in the sidebar, then click ",
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
                            "This dashboard analyzes cross-sectional momentum strategies across ETFs.",
                            className="card-text",
                        ),
                        html.Hr(),
                        html.H5("Quick Start"),
                        html.Ol(
                            [
                                html.Li("Select tickers in the sidebar (or use a preset)."),
                                html.Li("Adjust signal and portfolio parameters as needed."),
                                html.Li("Click Run Analysis."),
                                html.Li("Review charts, tables, and KPIs."),
                                html.Li("Click the download icon above any chart to export data as Excel."),
                            ]
                        ),
                        html.Hr(),
                        html.H5("Strategy Mode"),
                        html.Ul(
                            [
                                html.Li(
                                    [
                                        html.Strong("Single: "),
                                        "runs a pure momentum rotation strategy.",
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.Strong("Core-Satellite: "),
                                        "blends a fixed-weight core allocation with a momentum satellite. "
                                        "Configure core tickers/weights and the core ratio, then click Apply.",
                                    ]
                                ),
                            ]
                        ),
                        html.Hr(),
                        html.H5("Signal Parameters"),
                        html.P(
                            [
                                "Momentum is measured using three lookback periods (Short, Mid, Long). ",
                                "Assets are ranked by returns for each period, and the final signal is the weighted average of these ranks.",
                            ]
                        ),
                        html.Hr(),
                        html.H5("Portfolio Settings"),
                        html.Ul(
                            [
                                html.Li(
                                    [
                                        html.Strong("Top-K: "),
                                        "selects the top K ranked assets for investment.",
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.Strong("Quantile: "),
                                        "divides assets into groups to compare momentum quantiles.",
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.Strong("Weighting: "),
                                        "Equal (1/N), Inverse Vol, or Rank-based weights.",
                                    ]
                                ),
                            ]
                        ),
                        html.Hr(),
                        html.H5("Benchmark"),
                        html.Ul(
                            [
                                html.Li(
                                    [
                                        html.Strong("Equal Weight: "),
                                        "equal-weight basket of all selected tickers.",
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.Strong("Custom BM: "),
                                        "build a composite benchmark from multiple tickers with custom weights. "
                                        "Weights < 100% imply cash, > 100% imply leverage.",
                                    ]
                                ),
                            ]
                        ),
                        html.Hr(),
                        html.H5("Optimization"),
                        html.Ul(
                            [
                                html.Li(
                                    [
                                        html.Strong("Parameter Optimization: "),
                                        "grid search over signal windows and weights to maximize Sharpe ratio.",
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.Strong("Walk-Forward Analysis: "),
                                        "out-of-sample validation using rolling or expanding windows. "
                                        "Measures Sharpe decay to assess overfitting risk.",
                                    ]
                                ),
                            ]
                        ),
                        html.Hr(),
                        html.H5("Data Export"),
                        html.P(
                            "Click the download icon above NAV, drawdown, annual returns, monthly returns, "
                            "or holdings charts to export the underlying data as .xlsx."
                        ),
                    ]
                ),
                className="mb-4",
            ),
        ],
    )


def quick_reference_section():
    """Collapsible quick reference shown above the results."""
    return [
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
                        html.H6("Strategy Mode"),
                        html.Ul(
                            [
                                html.Li("Single: pure momentum rotation."),
                                html.Li("Core-Satellite: fixed core + momentum satellite, blended at weight level."),
                            ]
                        ),
                        html.H6("Signal Parameters"),
                        html.Ul(
                            [
                                html.Li("Signal Window: Lookback period (months) for Short, Mid, Long."),
                                html.Li("Signal Weight: Relative importance of each window (equal 1/3 or custom)."),
                            ]
                        ),
                        html.H6("Strategy Types"),
                        html.Ul(
                            [
                                html.Li("Top-K selects the top K ranked assets."),
                                html.Li("Quantile divides assets into groups to compare momentum."),
                            ]
                        ),
                        html.H6("Weighting Methods"),
                        html.Ul(
                            [
                                html.Li("Equal: 1/N weights."),
                                html.Li("Inverse Vol: higher weights to lower-volatility assets."),
                                html.Li("Rank: higher weights to better-ranked assets."),
                            ]
                        ),
                        html.H6("Benchmark"),
                        html.Ul(
                            [
                                html.Li("Equal Weight: equal-weight basket of selected tickers."),
                                html.Li("Custom BM: composite from multiple tickers with custom weights."),
                            ]
                        ),
                        html.H6("Optimization"),
                        html.Ul(
                            [
                                html.Li("Parameter Optimization: grid search to maximize Sharpe ratio."),
                                html.Li("Walk-Forward: out-of-sample validation with Sharpe decay analysis."),
                            ]
                        ),
                        html.H6("Data Export"),
                        html.Ul(
                            [
                                html.Li("Download icon above each chart exports data as .xlsx."),
                                html.Li("Available: NAV, drawdown, annual returns, monthly returns, holdings."),
                            ]
                        ),
                    ]
                ),
                className="mb-4",
            ),
        ),
    ]


def _kpi_section():
    """KPI cards and backtest period."""
    return [
        html.H4("Key Performance Indicators", className="mb-3"),
        html.Div(id="backtest-period-info", className="text-muted mb-3"),
        html.Div(id="kpi-cards-container", className="mb-4"),
    ]


def _performance_charts_section():
    """NAV, drawdown and annual returns charts."""
    return [
        html.H4("Performance Charts", className="mb-3"),
        html.Div(
            dbc.Button(html.I(className="fas fa-download"), id="dl-nav-btn",
                       color="secondary", outline=True, size="sm"),
            className="text-end mb-1",
        ),
        dcc.Graph(id="nav-chart", config={"displaylogo": False, "responsive": True}),
        html.Div(
            dbc.Button(html.I(className="fas fa-download"), id="dl-dd-btn",
                       color="secondary", outline=True, size="sm"),
            className="text-end mb-1",
        ),
        dcc.Graph(id="drawdown-chart", config={"displaylogo": False, "responsive": True}),

        # Annual Returns Bar Chart
        html.Div(
            dbc.Button(html.I(className="fas fa-download"), id="dl-annual-btn",
                       color="secondary", outline=True, size="sm"),
            className="text-end mb-1",
        ),
        dcc.Graph(id="annual-returns-chart", config={"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}),
    ]


def _monthly_returns_section():
    """Monthly returns table with series selector."""
    return [
        html.H4("Monthly Returns", className="mb-3"),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="monthly-returns-select",
                        options=[],
                        value=None,
                        clearable=False,
                        placeholder="Select strategy...",
                    ),
                    width=4,
                ),
                dbc.Col(
                    html.Div(
                        dbc.Button(html.I(className="fas fa-download"), id="dl-monthly-btn",
                                   color="secondary", outline=True, size="sm"),
                        className="text-end",
                    ),
                    width={"size": "auto", "offset": 0},
                    className="d-flex align-items-end ms-auto",
                ),
            ],
            className="mb-3",
        ),
        dcc.Graph(id="monthly-returns-chart", config={"displaylogo": False, "responsive": True}),

        html.Div(className="mb-4"),
    ]


def _signal_section():
    """Current signal, signal history and quantile spread."""
    return [
        html.H4("Current Signal by Quantile", className="mb-3"),
        html.Div(id="signal-date-info", className="text-muted mb-2"),
        html.Small(
            "Rank 1 represents the best-performing group, while higher ranks indicate worse performance. Entries are shown as Ticker (Rank).",
            className="text-muted d-block mb-2",
        ),
        html.Small(
            "If the latest month is not complete yet, the signal is preliminary — computed from the latest intra-month prices.",
            className="text-muted d-block mb-2",
        ),
        dcc.Graph(id="signal-table-chart", config={"displaylogo": False, "responsive": True}),

        # Signal History Collapse
        dbc.Button(
            "Show Signal History",
            id="raw-signal-collapse-btn",
            color="link",
            className="mb-2 p-0",
        ),
        dbc.Collapse(
            id="raw-signal-collapse",
            is_open=False,
            children=[
                html.H5("Signal History", className="mt-2"),
                html.Div(id="raw-signal-table-container"),
            ],
            className="mb-4",
        ),

        # Quantile Spread (only for Quantile strategy)
        html.Div(
            id="quantile-spread-section",
            children=[
                html.H4("Top–Bottom Quantile Spread", className="mb-3"),
                dcc.Graph(id="spread-chart", config={"displaylogo": False, "responsive": True}),
            ],
        ),
    ]


def _holdings_section():
    """Holdings heatmap with quantile / label selectors."""
    return [
        html.Div(className="mb-4"),
        html.H4("Holdings Analysis", className="mb-3"),
        html.Small(
            "The top quantile represents the best-performing assets, while the bottom quantile represents the worst.",
            className="text-muted d-block mb-2",
        ),
        html.Small(
            "Portfolio formed from each month-end signal and held for the following month. Each date is the start of that holding month.",
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
        dbc.Row(
            [
                dbc.Col(
                    dbc.RadioItems(
                        id="heatmap-label-toggle",
                        options=[
                            {"label": "Ticker", "value": "ticker"},
                            {"label": "Name", "value": "name"},
                        ],
                        value="ticker",
                        inline=True,
                        className="mb-0",
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button(html.I(className="fas fa-download"), id="dl-holdings-btn",
                               color="secondary", outline=True, size="sm"),
                    width="auto",
                    className="ms-auto",
                ),
            ],
            className="mb-1 align-items-center",
        ),
        dcc.Graph(id="heatmap-chart", config={"displaylogo": False, "responsive": True}),
    ]


def _statistics_section():
    """Statistics table and DataFrame."""
    return [
        html.H4("Statistics", className="mb-3"),
        dcc.Graph(id="stats-table-chart", config={"displaylogo": False, "responsive": True}),

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
    ]


def _walk_forward_results_section():
    """Walk-forward results (hidden until a run)."""
    return [
        html.Div(
            id="walk-forward-results-section",
            style={"display": "none"},
            children=[
                html.Hr(),
                html.Div(
                    [
                        html.H4(id="wf-results-title", className="d-inline me-3"),
                        dbc.Button(
                            [html.I(id="wf-collapse-icon", className="fas fa-chevron-up me-1"), "Hide"],
                            id="wf-results-collapse-btn",
                            color="link",
                            size="sm",
                            className="p-0",
                        ),
                    ],
                    className="d-flex align-items-center mb-3",
                ),
                dbc.Collapse(
                    id="wf-results-collapse",
                    is_open=True,
                    children=[
                        # WF Summary Metrics
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H6("In-Sample Sharpe (avg)", className="text-muted"),
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
                                                html.H6("Out-of-Sample Sharpe (avg)", className="text-muted"),
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
                                                html.H6("Combined OOS Sharpe", className="text-muted"),
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
                                                        html.H6("Windows (S/M/L)", className="text-muted"),
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
                                                        html.H6("Weights (S/M/L)", className="text-muted"),
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
                                                        html.H6("In-Sample Sharpe", className="text-muted"),
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
                                    [html.I(className="fas fa-check me-2"), "Apply Recommended Params"],
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
                        dcc.Graph(id="wf-oos-nav-chart", config={"displaylogo": False, "responsive": True}),
                        html.Div(id="wf-oos-sharpe-comparison", className="text-muted mb-3"),
                    ],
                ),
            ],
        ),
    ]


def create_main_content():
    """Create the main content area."""
    return dbc.Col(
        [
            html.H2("Trend Rotation Strategy Dashboard", className="mb-4"),

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

            create_instructions(),

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
                            *_kpi_section(),
                            *quick_reference_section(),
                            *_performance_charts_section(),
                            *_monthly_returns_section(),
                            *_signal_section(),
                            *_holdings_section(),
                            *_statistics_section(),
                            *_walk_forward_results_section(),
                            # Bottom spacing
                            html.Div(style={"height": "100px"}),
                        ],
                    ),
                ],
            ),
        ],
        lg=9,
        xs=12,
        className="main-content p-3 p-lg-4",
    )
