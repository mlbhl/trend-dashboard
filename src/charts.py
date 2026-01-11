"""Chart generation functions using Plotly."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_nav_chart(
    navs: pd.DataFrame,
    title: str = "Portfolio NAV",
    height: int = 500,
) -> go.Figure:
    """
    Create NAV time series chart.

    Args:
        navs: NAV DataFrame
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, col in enumerate(navs.columns):
        fig.add_trace(go.Scatter(
            x=navs.index,
            y=navs[col],
            mode='lines',
            name=col,
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="NAV",
        height=height,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    return fig


def create_drawdown_chart(
    navs: pd.DataFrame,
    title: str = "Drawdown",
    height: int = 300,
) -> go.Figure:
    """
    Create drawdown chart.

    Args:
        navs: NAV DataFrame
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure
    """
    cummax = navs.cummax()
    drawdown = (navs - cummax) / cummax * 100  # In percentage

    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, col in enumerate(drawdown.columns):
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown[col],
            mode='lines',
            name=col,
            fill='tozeroy',
            line=dict(color=colors[i % len(colors)], width=1),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=height,
        hovermode='x unified',
        showlegend=False,
    )

    return fig


def create_allocation_chart(
    weight: pd.DataFrame,
    title: str = "Portfolio Allocation",
    height: int = 400,
) -> go.Figure:
    """
    Create stacked area chart for portfolio allocation.

    Args:
        weight: Portfolio weights DataFrame
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure
    """
    # Filter to non-zero columns
    active_cols = weight.columns[weight.sum() > 0]
    weight_active = weight[active_cols]

    fig = go.Figure()

    colors = px.colors.qualitative.Set3

    for i, col in enumerate(weight_active.columns):
        fig.add_trace(go.Scatter(
            x=weight_active.index,
            y=weight_active[col],
            mode='lines',
            name=col,
            stackgroup='one',
            line=dict(width=0),
            fillcolor=colors[i % len(colors)],
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Weight",
        height=height,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    return fig


def create_holding_heatmap(
    weight: pd.DataFrame,
    title: str = "Monthly Holdings",
    height: int = 500,
) -> go.Figure:
    """
    Create heatmap for monthly holdings.

    Args:
        weight: Portfolio weights DataFrame
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure
    """
    
    w = weight.copy()
    w.index = pd.to_datetime(w.index)

    w = w.sort_index()

    weight_t = w.T
    x_dates = pd.to_datetime(weight_t.columns)

    step = 3
    tick_idx = list(range(0, len(x_dates), step))
    tickvals = x_dates[tick_idx]
    ticktext = [d.strftime("%y.%m") for d in tickvals]

    fig = go.Figure(data=go.Heatmap(
        z=weight_t.values,
        x=x_dates,
        y=weight_t.index,
        colorscale='Blues',
        xgap=0.1,
        ygap=0.1,
        hovertemplate='Ticker: %{y}<br>Date: %{x}<br>Weight: %{z:.2%}<extra></extra>',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Ticker",
        height=height,
        xaxis=dict(
            type="date",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=45,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            autorange="reversed",
        ),
    )

    return fig


def create_signal_category_table(
    signal: pd.Series,
    n_quantiles: int = 5,
    title: str = "Current Signal by Quantile",
) -> go.Figure:
    """
    Create a table showing tickers grouped by quantile category.
    Rank 1 is the best (goes to Q5).

    Args:
        signal: Series with ticker rankings (1 = best)
        n_quantiles: Number of quantile groups
        title: Chart title

    Returns:
        Plotly Figure
    """
    import numpy as np

    signal = signal.dropna()
    n = len(signal)

    # Assign quantiles: rank 1 (best) -> Q5, worst rank -> Q1
    # Match portfolio.py logic: ascending=False means lower signal values get higher ranks
    quantile_labels = pd.qcut(
        signal.rank(method="first", ascending=False),
        q=n_quantiles,
        labels=[f"Q{i}" for i in range(1, n_quantiles + 1)]
    )

    # Group tickers by quantile
    quantile_data = {}
    for q in range(n_quantiles, 0, -1):
        q_label = f"Q{q}"
        tickers_in_q = quantile_labels[quantile_labels == q_label].index.tolist()
        # Sort by rank within quantile
        tickers_sorted = signal.loc[tickers_in_q].sort_values()
        quantile_data[q_label] = [
            f"{t} ({int(signal[t])})" for t in tickers_sorted.index
        ]

    # Find max length for padding
    max_len = max(len(v) for v in quantile_data.values())

    # Pad shorter lists with empty strings
    for q in quantile_data:
        while len(quantile_data[q]) < max_len:
            quantile_data[q].append("")

    # Create columns in order Q5 -> Q1 (best to worst)
    headers = [f"Q{q}" for q in range(n_quantiles, 0, -1)]
    cell_values = [quantile_data[h] for h in headers]

    # Color headers: Q5 (best) is green, Q1 (worst) is red
    header_colors = []
    for i, q in enumerate(range(n_quantiles, 0, -1)):
        ratio = (q - 1) / (n_quantiles - 1)  # 1 for Q5, 0 for Q1
        r = int(255 * (1 - ratio))
        g = int(200 * ratio)
        header_colors.append(f'rgb({r}, {g}, 100)')

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color=header_colors,
            font=dict(color='white', size=14, family='Arial Black'),
            align='center',
            height=35,
        ),
        cells=dict(
            values=cell_values,
            fill_color='white',
            font=dict(color='black', size=12),
            align='center',
            height=28,
        ),
    )])

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=max(250, 40 + max_len * 30),
    )

    return fig


def create_quantile_spread_chart(
    q_nav: pd.DataFrame,
    title: str = "Q1 vs Q5 Spread",
    height: int = 400,
) -> go.Figure:
    """
    Create chart showing spread between top and bottom quantiles.

    Args:
        q_nav: Quantile NAV DataFrame
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure
    """
    if 'Q1' not in q_nav.columns or 'Q5' not in q_nav.columns:
        return go.Figure()

    spread = q_nav['Q5'] - q_nav['Q1']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=spread.index,
        y=spread,
        mode='lines',
        name='Q5 - Q1 Spread',
        line=dict(color='darkblue', width=2),
        fill='tozeroy',
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Spread (NAV)",
        height=height,
    )

    return fig


def create_returns_table(stats: pd.DataFrame) -> go.Figure:
    """
    Create a formatted table for performance statistics.

    Args:
        stats: Statistics DataFrame from summary_stats

    Returns:
        Plotly Figure
    """
    # Format values
    formatted = stats.copy()

    # Format specific rows
    pct_rows = ['cagr', 'mean', 'vol', 'max', 'min', 'mdd']
    for row in pct_rows:
        if row in formatted.index:
            formatted.loc[row] = formatted.loc[row].apply(lambda x: f'{x:.2%}')

    ratio_rows = ['sharpe', 'skew', 'kurt', 'mean-t-stat']
    for row in ratio_rows:
        if row in formatted.index:
            formatted.loc[row] = formatted.loc[row].apply(lambda x: f'{x:.2f}')

    if 'nyears' in formatted.index:
        formatted.loc['nyears'] = formatted.loc['nyears'].apply(lambda x: f'{x:.1f}')
    if 'nsamples' in formatted.index:
        formatted.loc['nsamples'] = formatted.loc['nsamples'].apply(lambda x: f'{int(x)}')

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Metric'] + list(formatted.columns),
            fill_color='darkblue',
            font=dict(color='white', size=12),
            align='left',
            height=30,
        ),
        cells=dict(
            values=[formatted.index] + [formatted[col] for col in formatted.columns],
            fill_color='lavender',
            font=dict(color='black', size=12),
            align='left',
            height=25,
        ),
    )])

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig
