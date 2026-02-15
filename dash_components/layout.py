"""Layout definition for Trend Dashboard Dash app."""

from datetime import date
import dash_bootstrap_components as dbc
from dash import dcc, html

from src.config import (
    TICKER_PRESETS,
    DEFAULT_PRESET,
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


def _create_core_row(index: int):
    """Create a single core ticker/weight row for the Core-Satellite UI."""
    return html.Div(
        id={"type": "core-row", "index": index},
        children=[
            dbc.InputGroup(
                [
                    dbc.Input(
                        id={"type": "core-row-ticker", "index": index},
                        placeholder="e.g., SPY",
                        type="text",
                        value="",
                        style={"flex": "2"},
                    ),
                    dbc.Input(
                        id={"type": "core-row-weight", "index": index},
                        placeholder="%",
                        type="number",
                        min=0,
                        max=100,
                        step="any",
                        value=100,
                        style={"flex": "1"},
                    ),
                    dbc.InputGroupText("%", style={"padding": "0.25rem 0.5rem"}),
                    dbc.Button(
                        html.I(className="fas fa-times"),
                        id={"type": "core-row-delete", "index": index},
                        color="danger",
                        outline=True,
                        size="sm",
                        style={"padding": "0.25rem 0.5rem"},
                    ),
                ],
                size="sm",
                className="mb-1",
            ),
        ],
    )


def _create_bm_row(index: int):
    """Create a single benchmark ticker/weight row for the Custom BM UI."""
    return html.Div(
        id={"type": "bm-row", "index": index},
        children=[
            dbc.InputGroup(
                [
                    dbc.Input(
                        id={"type": "bm-row-ticker", "index": index},
                        placeholder="e.g., SPY",
                        type="text",
                        style={"flex": "2"},
                    ),
                    dbc.Input(
                        id={"type": "bm-row-weight", "index": index},
                        placeholder="%",
                        type="number",
                        min=0,
                        max=200,
                        step="any",
                        value=100,
                        style={"flex": "1"},
                    ),
                    dbc.InputGroupText("%", style={"padding": "0.25rem 0.5rem"}),
                    dbc.Button(
                        html.I(className="fas fa-times"),
                        id={"type": "bm-row-delete", "index": index},
                        color="danger",
                        outline=True,
                        size="sm",
                        style={"padding": "0.25rem 0.5rem"},
                    ),
                ],
                size="sm",
                className="mb-1",
            ),
        ],
    )


def create_sidebar_content():
    """Create the sidebar content (controls only, no wrapper)."""
    return [
        html.H4("Parameters", className="mb-3"),

            # Strategy Mode Section
            html.H6("Strategy Mode", className="text-muted mb-2"),
            dbc.RadioItems(
                id="strategy-mode-radio",
                options=[
                    {"label": "Single", "value": "single"},
                    {"label": "Core-Satellite", "value": "core-satellite"},
                ],
                value="single",
                inline=True,
                className="mb-2",
            ),
            html.Div(
                id="core-satellite-div",
                style={"display": "none"},
                children=[
                    label_with_help(
                        "Core Allocation",
                        "help-core-alloc",
                        "Percentage of total portfolio allocated to the core (fixed-weight) component. The remainder goes to the satellite (momentum) strategy.",
                    ),
                    dbc.InputGroup(
                        [
                            dbc.Input(
                                id="core-weight-input",
                                type="number",
                                min=1,
                                max=99,
                                step=1,
                                value=70,
                            ),
                            dbc.InputGroupText("%"),
                        ],
                        size="sm",
                        className="mb-1",
                    ),
                    html.Small(
                        id="satellite-weight-label",
                        children="Satellite: 30%",
                        className="text-muted d-block mb-2",
                    ),
                    label_with_help(
                        "Core Tickers & Weights",
                        "help-core-tickers",
                        "Select tickers and their relative weights within the core allocation. Weights are normalized to sum to 100%.",
                    ),
                    html.Div(
                        id="core-rows-container",
                        children=[_create_core_row(0)],
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-plus me-1"), "Add"],
                        id="core-add-row-btn",
                        color="secondary",
                        outline=True,
                        size="sm",
                        className="mb-2",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-check me-1"), "Apply"],
                        id="apply-core-btn",
                        color="success",
                        outline=True,
                        size="sm",
                        className="mb-2 ms-2",
                    ),
                    html.Div(id="core-ticker-status", className="mb-2"),
                ],
            ),
            dcc.Store(id="core-config-store", data=[]),
            html.Hr(),

            # Ticker Selection Section
            html.H6("Ticker Selection", className="text-muted mb-2"),
            label_with_help("Preset", "help-preset", "Select a predefined ticker set. You can add or remove tickers after selection."),
            dcc.Dropdown(
                id="ticker-preset-select",
                options=[{"label": k, "value": k} for k in TICKER_PRESETS.keys()],
                value=DEFAULT_PRESET,
                clearable=False,
                className="mb-2",
            ),
            label_with_help("Tickers", "help-tickers", "Select ETFs to include in the analysis. Click X to remove."),
            dcc.Store(id="ticker-store", data=TICKER_PRESETS[DEFAULT_PRESET].copy()),
            html.Div(id="ticker-tags", className="mb-2", style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
            label_with_help("Add Ticker", "help-add-ticker", "Enter a ticker symbol and press Enter to add."),
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
            html.Div(
                [
                    dbc.Button(
                        "Reset to Preset",
                        id="reset-tickers-btn",
                        color="link",
                        size="sm",
                        className="p-0 me-3",
                    ),
                    dbc.Button(
                        "Clear All",
                        id="clear-tickers-btn",
                        color="link",
                        size="sm",
                        className="p-0",
                    ),
                ],
                className="d-flex mb-2",
            ),
            label_with_help("Min Valid Tickers", "help-thresh", "Minimum number of valid tickers required for the analysis."),
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

            # Data Settings Section
            html.H6("Data Settings", className="text-muted mb-2"),
            label_with_help("Download Start Date", "help-start-date", "Start date for downloading historical data from Yahoo Finance."),
            dcc.DatePickerSingle(
                id="start-date",
                date=DEFAULT_START_DATE,
                min_date_allowed="1990-01-01",
                max_date_allowed=date.today(),
                display_format="YYYY-MM-DD",
                className="mb-2 w-100",
            ),
            label_with_help("Backtest Start Date", "help-backtest-date", "If insufficient data are available, the earliest available date will be used."),
            dcc.DatePickerSingle(
                id="backtest-start-date",
                date=DEFAULT_BACKTEST_START_DATE,
                min_date_allowed="1990-01-01",
                max_date_allowed=date.today(),
                display_format="YYYY-MM-DD",
                className="mb-3 w-100",
            ),
            html.Hr(),

            # Signal Parameters Section (Collapsible)
            html.Div(
                [
                    html.H6("Signal Parameters", className="text-muted d-inline me-2"),
                    html.Span(
                        html.I(className="fas fa-question-circle text-muted"),
                        id="help-signal-params",
                        style={"cursor": "pointer", "fontSize": "0.8rem"},
                    ),
                    dbc.Tooltip(
                        "Parameters for momentum signal calculation including lookback windows and their weights.",
                        target="help-signal-params",
                        placement="right",
                    ),
                ],
                className="mb-2",
            ),
            dbc.Button(
                "Signal Settings",
                id="signal-settings-collapse-btn",
                color="link",
                size="sm",
                className="p-0 mb-2",
            ),
            dbc.Collapse(
                id="signal-settings-collapse",
                is_open=False,
                children=[
                    # Signal Window Parameters
                    html.Div(
                        label_with_help("Window Parameters", "help-signal-windows", "Lookback periods for momentum calculation. Shorter windows are more reactive, while longer windows are more stable."),
                        className="mt-2",
                    ),
                    dbc.Label("Short Window (months)", size="sm"),
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
                    dbc.Label("Mid Window (months)", size="sm"),
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
                    dbc.Label("Long Window (months)", size="sm"),
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

                    # Signal Weight Parameters
                    label_with_help("Weight Parameters", "help-signal-weights", "Relative importance of each window. The final signal is computed as the weighted average of three momentum ranks."),
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
                            dbc.Label("Short Weight", size="sm"),
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
                            dbc.Label("Mid Weight", size="sm"),
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
                            dbc.Label("Long Weight", size="sm"),
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
                ],
            ),
            html.Hr(),

            # Portfolio Settings Section
            html.H6("Portfolio Settings", className="text-muted mb-2"),
            label_with_help("Strategy Type", "help-strategy-type", "Top-K selects the top K ranked assets for investment. Quantile divides assets into groups to analyze momentum."),
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
                        className="me-2",
                    ),
                    html.Span(
                        html.I(className="fas fa-question-circle text-muted"),
                        id="help-select-all",
                        style={"cursor": "pointer", "fontSize": "0.8rem"},
                    ),
                    dbc.Tooltip(
                        "Select all available tickers at each rebalancing. This option is only meaningful with Rank weighting, where the signal determines portfolio weights.",
                        target="help-select-all",
                        placement="right",
                    ),
                ],
                className="d-flex align-items-center mb-2",
            ),
            html.Div(
                id="top-k-div",
                children=[
                    label_with_help("Top K", "help-top-k", "Number of top-ranked assets to include in the portfolio. Lower K results in a more concentrated portfolio."),
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
            label_with_help("Number of Quantiles", "help-n-quantiles", "Number of groups to divide assets into. For five quantiles, Q5 represents the best-ranked group and Q1 the worst."),
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
            label_with_help("Weighting Method", "help-weight-method", 
                            "Portfolio weighting scheme. Equal assigns 1/N weights, Inverse Vol weights assets by the inverse of volatility, and Rank weights assets by the inverse of rank."),
            dcc.Dropdown(
                id="weight-method-select",
                options=[{"label": k, "value": v} for k, v in WEIGHT_METHODS.items()],
                value="equal",
                clearable=False,
                className="mb-2",
            ),
            html.Hr(),

            # Transaction Cost Section
            html.H6("Transaction Costs", className="text-muted mb-2"),
            label_with_help("One-way Transaction Cost per Trade", "help-tcost", "In %. 5bp = 0.05%. Applied to both buys and sells."),
            dbc.InputGroup(
                [
                    dbc.Input(
                        id="tcost-input",
                        type="number",
                        min=0,
                        max=1.0,
                        step=0.01,
                        value=0.00,
                    ),
                    dbc.InputGroupText("%"),
                ],
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
                            {"label": "Custom BM", "value": "custom"},
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
                        "Equal Weight assigns equal weights across all selected tickers. Custom BM builds a composite benchmark from multiple tickers with custom weights.",
                        target="help-bm-type",
                        placement="right",
                    ),
                ],
                className="mb-2",
            ),
            dcc.Store(id="bm-config-store", data=[]),
            html.Div(
                id="custom-bm-div",
                children=[
                    html.Div(
                        id="bm-rows-container",
                        children=[
                            _create_bm_row(0),
                        ],
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-plus me-1"), "Add"],
                        id="bm-add-row-btn",
                        color="secondary",
                        outline=True,
                        size="sm",
                        className="mb-2",
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-check me-1"), "Apply"],
                        id="apply-bm-btn",
                        color="success",
                        outline=True,
                        size="sm",
                        className="mb-2 ms-2",
                    ),
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
                        "Uses grid search to optimize window and weight parameters by maximizing the Sharpe ratio over the backtest period.",
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
                        "Full Grid evaluates all 3,696 parameter combinations. Random Grid samples a subset to reduce computation time.",
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
                                    label_with_help("Samples", "help-opt-samples", "Number of random parameter combinations to test (100 to 3,696).", size="sm"),
                                    dbc.Input(
                                        id="opt-samples-input",
                                        type="number",
                                        value=500,
                                        min=1,
                                        max=3696,
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
                        "Performs out-of-sample validation by training on past data and testing on future data. Assesses strategy robustness and overfitting risk.",
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
                                "Rolling uses a fixed lookback window. Expanding uses all available historical data.",
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
                    label_with_help("Train Period (months)", "help-wf-train", "Length of the in-sample training period. Longer periods provide more data but fewer folds.", size="sm"),
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
                    label_with_help("Test Period (months)", "help-wf-test", "Length of the out-of-sample test period for each fold.", size="sm"),
                    dbc.Input(
                        id="wf-test-input",
                        type="number",
                        value=12,
                        min=1,
                        step=1,
                        size="sm",
                        className="mb-2",
                    ),
                    label_with_help("Step Size (months)", "help-wf-step", "How far the window moves forward between folds. Smaller step sizes lead to more overlap.", size="sm"),
                    dbc.Input(
                        id="wf-step-input",
                        type="number",
                        value=12,
                        min=1,
                        step=1,
                        size="sm",
                        className="mb-2",
                    ),
                    label_with_help("Samples per Fold", "help-wf-samples", "Number of random parameter combinations to test per fold (100 to 3,696).", size="sm"),
                    dbc.Input(
                        id="wf-samples-input",
                        type="number",
                        value=500,
                        min=1,
                        max=3696,
                        step=1,
                        size="sm",
                        className="mb-1",
                    ),
                    html.Div(id="wf-samples-warning", className="mb-1"),
                    label_with_help("Seed", "help-wf-seed", "Random seed for reproducible results.", size="sm"),
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

            # Copyright
            html.Hr(className="mt-4"),
            html.Small(
                "© 2026 Byounghyo Lim. All rights reserved.",
                className="text-muted d-block text-center",
            ),
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

            # Initial instructions (shown when no data)
            html.Div(
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
                                        html.Li("Select tickers in the sidebar (or use the default set)."),
                                        html.Li("Click Run Analysis."),
                                        html.Li("Review the results."),
                                    ]
                                ),
                                html.Hr(),
                                html.H5("Signal Parameters"),
                                html.P(
                                    [
                                        "Momentum is measured using three lookback periods (Short, Mid, and Long).",
                                        html.Br(),
                                        "For each period, assets are ranked by returns, and the final signal is computed as the weighted average of these ranks.",
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
                                                "divides assets into groups to analyze momentum.",
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
                                                    html.Li("Signal Window: Lookback period (in months) for each timeframe."),
                                                    html.Li("Signal Weight: Weight assigned to each window (equal 1/3 or custom)."),
                                                ]
                                            ),
                                            html.H6("Strategy Types"),
                                            html.Ul(
                                                [
                                                    html.Li("Top-K selects the top K ranked assets for investment."),
                                                    html.Li("Quantile divides assets into groups to analyze momentum."),
                                                ]
                                            ),
                                            html.H6("Weighting Methods"),
                                            html.Ul(
                                                [
                                                    html.Li("Equal assigns 1/N weights to each asset."),
                                                    html.Li("Inverse Vol assigns higher weights to lower-volatility assets."),
                                                    html.Li("Rank assigns higher weights to better-ranked assets."),
                                                ]
                                            ),
                                        ]
                                    ),
                                    className="mb-4",
                                ),
                            ),

                            # Performance Charts Section
                            html.H4("Performance Charts", className="mb-3"),
                            dcc.Graph(id="nav-chart", config={"displaylogo": False, "responsive": True}),
                            dcc.Graph(id="drawdown-chart", config={"displaylogo": False, "responsive": True}),

                            # Signal Category Table
                            html.H4("Current Signal by Quantile", className="mb-3"),
                            html.Div(id="signal-date-info", className="text-muted mb-2"),
                            html.Small(
                                "Rank 1 represents the best-performing group, while higher ranks indicate worse performance. Entries are shown as Ticker (Rank).",
                                className="text-muted d-block mb-2",
                            ),
                            dcc.Graph(id="signal-table-chart", config={"displaylogo": False, "responsive": True}),

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
                                    html.H4("Top–Bottom Quantile Spread", className="mb-3"),
                                    dcc.Graph(id="spread-chart", config={"displaylogo": False, "responsive": True}),
                                ],
                            ),

                            # Holdings Heatmap Section
                            html.H4("Holdings Analysis", className="mb-3"),
                            html.Small(
                                "The top quantile represents the best-performing assets, while the bottom quantile represents the worst.",
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
                            dcc.Graph(id="heatmap-chart", config={"displaylogo": False, "responsive": True}),

                            # Performance Statistics Table
                            html.H4("Performance Statistics", className="mb-3"),
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

                            # Walk-Forward Results Section
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
