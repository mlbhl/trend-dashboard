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
        autosize=True,
        hovermode='x unified',
        dragmode=False,
        xaxis=dict(fixedrange=True, hoverformat="%Y-%m-%d"),
        yaxis=dict(fixedrange=True),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=20, t=40, b=40),
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
        autosize=True,
        hovermode='x unified',
        dragmode=False,
        xaxis=dict(fixedrange=True, hoverformat="%Y-%m-%d"),
        yaxis=dict(fixedrange=True),
        margin=dict(l=40, r=20, t=40, b=40),
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
        autosize=True,
        hovermode='x unified',
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=20, t=40, b=40),
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
        hovertemplate='Ticker: %{y}<br>Date: %{x|%y.%m}<br>Weight: %{z:.2%}<extra></extra>',
        showscale=False,
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Ticker",
        height=height,
        autosize=True,
        dragmode=False,
        xaxis=dict(
            type="date",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=45,
            fixedrange=True,
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            autorange="reversed",
            fixedrange=True,
        ),
        margin=dict(l=60, r=20, t=40, b=60),
    )

    return fig


def create_quantile_holding_heatmap(
    weights_dict: dict[str, pd.DataFrame],
    title: str = "Monthly Holdings by Quantile",
    height: int = 500,
    ticker_descriptions: dict[str, str] | None = None,
    use_name: bool = False,
    signal: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Create heatmap showing all quantiles at once with different colors.

    Args:
        weights_dict: Dict mapping quantile name (e.g. "Q1") to weight DataFrame
        title: Chart title
        height: Chart height in pixels
        ticker_descriptions: Dict mapping ticker to description name
        use_name: If True, show description names on y-axis instead of tickers
        signal: Signal DataFrame with rankings (rank 1 = best). Used for y-axis sorting.

    Returns:
        Plotly Figure
    """
    import numpy as np

    n_q = len(weights_dict)
    quantile_names = sorted(weights_dict.keys(), key=lambda x: int(x[1:]))

    # Collect all tickers and dates across all quantiles
    all_tickers = set()
    all_dates = None
    for wgt in weights_dict.values():
        all_tickers.update(wgt.columns)
        if all_dates is None:
            all_dates = wgt.index
    dates = pd.to_datetime(all_dates).sort_values()

    # Sort tickers by quantile (high Q first), then by signal score within quantile
    # Signal has rankings where rank 1 = best (lowest = better)
    last_date = dates[-1]
    ticker_q = {}
    for q_name in quantile_names:
        q_num = int(q_name[1:])
        wgt_df = weights_dict[q_name]
        wgt_df_idx = pd.to_datetime(wgt_df.index)
        if last_date in wgt_df_idx:
            row = wgt_df.loc[wgt_df.index[wgt_df_idx == last_date][0]]
            for ticker in row.index[row.abs() > 1e-10]:
                ticker_q[ticker] = q_num

    # Get latest signal ranking for sorting within quantile
    ticker_rank = {}
    if signal is not None and len(signal) > 0:
        last_signal = signal.iloc[-1]
        for ticker in all_tickers:
            if ticker in last_signal.index and pd.notna(last_signal[ticker]):
                ticker_rank[ticker] = last_signal[ticker]

    # Sort: high Q first, within same Q lower rank (better) first
    max_rank = len(all_tickers) + 1
    all_tickers = sorted(
        all_tickers,
        key=lambda t: (-ticker_q.get(t, 0), ticker_rank.get(t, max_rank)),
    )

    # Build y-axis labels
    desc = ticker_descriptions or {}
    if use_name:
        y_labels = [desc.get(t, t) for t in all_tickers]
    else:
        y_labels = list(all_tickers)

    # Build quantile assignment matrix (ticker x date) using vectorized ops
    quantile_matrix = pd.DataFrame(np.nan, index=all_tickers, columns=dates)
    weight_matrix = pd.DataFrame(0.0, index=all_tickers, columns=dates)

    for q_name in quantile_names:
        q_num = int(q_name[1:])
        wgt_df = weights_dict[q_name].copy()
        wgt_df.index = pd.to_datetime(wgt_df.index)
        wgt_t = wgt_df.T.reindex(index=all_tickers, columns=dates)
        held = wgt_t.abs() > 1e-10
        quantile_matrix[held] = q_num
        weight_matrix[held] = wgt_t[held]

    x_dates = dates
    step = 3
    tick_idx = list(range(0, len(x_dates), step))
    tickvals = x_dates[tick_idx]
    ticktext = [d.strftime("%y.%m") for d in tickvals]

    # Build custom colorscale — mid-tone, clear but not harsh
    _palette = [
        "#e06666",  # muted red (worst)
        "#f0a050",  # warm orange
        "#b8b8b8",  # medium gray
        "#6fbf6f",  # medium green
        "#6fa8dc",  # medium blue (best)
    ]
    if n_q <= len(_palette):
        # Pick evenly spaced colors: worst=red ... best=blue
        indices = [int(i * (len(_palette) - 1) / (n_q - 1)) for i in range(n_q)] if n_q > 1 else [2]
        colors = [_palette[i] for i in indices]
    else:
        colors = px.colors.qualitative.Set1[:n_q]
    colorscale = []
    for i in range(n_q):
        lo = i / n_q
        hi = (i + 1) / n_q
        colorscale.append([lo, colors[i]])
        colorscale.append([hi, colors[i]])

    # Custom hover text (always show both ticker and name)
    q_matrix_vals = quantile_matrix.values
    w_matrix_vals = weight_matrix.values
    hover_text = []
    for i, ticker in enumerate(all_tickers):
        name = desc.get(ticker, ticker)
        row = []
        for j, dt in enumerate(dates):
            q_val = q_matrix_vals[i, j]
            w_val = w_matrix_vals[i, j]
            if np.isnan(q_val):
                row.append("")
            else:
                row.append(
                    f"{ticker} ({name})<br>Date: {dt.strftime('%y.%m')}"
                    f"<br>Q{int(q_val)}<br>Weight: {w_val:.2%}"
                )
        hover_text.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=q_matrix_vals,
        x=x_dates,
        y=y_labels,
        colorscale=colorscale,
        zmin=1,
        zmax=n_q,
        xgap=0.5,
        ygap=0.5,
        hovertext=hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        showscale=False,
    ))

    # Build legend text inline with title (highest quantile first)
    legend_parts = []
    for q_name in reversed(quantile_names):
        q_num = int(q_name[1:])
        color = colors[q_num - 1]
        label = q_name
        if q_num == n_q:
            label += " (Best)"
        elif q_num == 1:
            label += " (Worst)"
        legend_parts.append(f"<span style='color:{color}'>\u25a0</span> {label}")
    legend_str = "&nbsp;&nbsp;".join(legend_parts)
    title_with_legend = f"{title}&nbsp;&nbsp;&nbsp;&nbsp;{legend_str}"

    fig.update_layout(
        title=dict(text=title_with_legend, font=dict(size=16)),
        xaxis_title="Month",
        yaxis_title="",
        height=height,
        autosize=True,
        dragmode=False,
        font=dict(size=13),
        xaxis=dict(
            type="date",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=45,
            tickfont=dict(size=14),
            fixedrange=True,
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            autorange="reversed",
            tickfont=dict(size=14),
            fixedrange=True,
        ),
        margin=dict(l=60, r=20, t=40, b=60),
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
            height=30,
        ),
    )])

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=100 + max_len * 50,
        autosize=True,
    )

    return fig


def create_quantile_spread_chart(
    q_nav: pd.DataFrame,
    top_q: str = "Q5",
    title: str = "Top vs Bottom Quantile Spread",
    height: int = 400,
) -> go.Figure:
    """
    Create chart showing spread between top and bottom quantiles.

    Args:
        q_nav: Quantile NAV DataFrame
        top_q: Top quantile column name (e.g. "Q3", "Q5", "Q10")
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure
    """
    if 'Q1' not in q_nav.columns or top_q not in q_nav.columns:
        return go.Figure()

    spread = q_nav[top_q] - q_nav['Q1']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=spread.index,
        y=spread,
        mode='lines',
        name=f'{top_q} - Q1 Spread',
        line=dict(color='lightblue', width=2),
        fill='tozeroy',
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Spread (NAV)",
        height=height,
        autosize=True,
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        margin=dict(l=40, r=20, t=40, b=40),
    )

    return fig


def create_annual_returns_chart(
    annual_rets: pd.DataFrame,
    title: str = "Annual Returns",
    height: int = 400,
) -> go.Figure:
    """Create grouped bar chart of annual returns.

    Args:
        annual_rets: DataFrame from metrics.annual_returns() (years x columns)
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure
    """
    import numpy as np

    colors = px.colors.qualitative.Set2
    fig = go.Figure()

    for i, col in enumerate(annual_rets.columns):
        vals = annual_rets[col].values
        fig.add_trace(go.Bar(
            x=[str(y) for y in annual_rets.index],
            y=vals,
            name=col,
            marker_color=colors[i % len(colors)],
            text=[f"{v:.1%}" for v in vals],
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate="Year: %{x}<br>" + col + ": %{y:.1%}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Return",
        yaxis_tickformat=".0%",
        barmode="group",
        height=height,
        autosize=True,
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
    )

    return fig


def create_monthly_returns_table(
    monthly_rets: pd.DataFrame,
    title: str = "Monthly Returns",
    height: int = None,
) -> go.Figure:
    """Create annotated heatmap of monthly returns (year x month).

    Args:
        monthly_rets: DataFrame from metrics.monthly_returns() (year x 1..12)
        title: Chart title
        height: Chart height in pixels (auto-calculated if None)

    Returns:
        Plotly Figure
    """
    import numpy as np

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    z = monthly_rets.values * 100  # convert to percentage for display
    years = [str(y) for y in monthly_rets.index]

    # Annotation text
    text = []
    for row in z:
        row_text = []
        for v in row:
            if np.isnan(v):
                row_text.append("")
            else:
                row_text.append(f"{v:.1f}%")
        text.append(row_text)

    if height is None:
        height = max(300, len(years) * 35 + 80)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=month_labels,
        y=years,
        colorscale="RdYlGn",
        zmid=0,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{text}<extra></extra>",
        showscale=False,
        xgap=1,
        ygap=1,
    ))

    fig.update_layout(
        title=title,
        height=height,
        autosize=True,
        dragmode=False,
        xaxis=dict(fixedrange=True, side="top"),
        yaxis=dict(fixedrange=True, autorange="reversed"),
        margin=dict(l=60, r=20, t=60, b=20),
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
    pct_rows = ['cumulative', 'cagr', 'mean', 'vol', 'max', 'min', 'mdd']
    for row in pct_rows:
        if row in formatted.index:
            formatted.loc[row] = formatted.loc[row].apply(lambda x: f'{x:.1%}')

    ratio_rows = ['sharpe', 'skew', 'kurt']
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
            font=dict(color='white', size=14),
            align='left',
            height=50,
        ),
        cells=dict(
            values=[formatted.index] + [formatted[col] for col in formatted.columns],
            fill_color='lavender',
            font=dict(color='black', size=14),
            align='left',
            height=30,
        ),
    )])

    # Calculate height based on number of rows
    n_rows = len(formatted.index)
    table_height = 50 + (n_rows * 30) + 20  # header + rows + padding

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=table_height,
    )

    return fig
