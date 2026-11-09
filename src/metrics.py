"""Performance metrics and statistics functions."""

import numpy as np
import pandas as pd


def summary_stats(navs: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """
    Calculate comprehensive performance statistics.

    Args:
        navs: NAV DataFrame or Series

    Returns:
        DataFrame with performance metrics
    """
    if isinstance(navs, pd.Series):
        navs = navs.to_frame()

    # Remove NaN rows
    navs = navs.dropna()

    # Calculate returns
    returns = navs.pct_change().dropna()

    # Time period
    n_samples = len(returns)
    n_years = n_samples / 252  # Trading days per year

    # Basic return statistics
    total_return = navs.iloc[-1] / navs.iloc[0] - 1
    cagr = (navs.iloc[-1] / navs.iloc[0]) ** (1 / n_years) - 1
    mean_return = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)

    # Risk-adjusted
    sharpe = mean_return / vol

    # Drawdown
    cummax = navs.cummax()
    drawdown = (navs - cummax) / cummax
    mdd = drawdown.min()

    # Higher moments
    skew = returns.skew()
    kurt = returns.kurtosis()


    stats = pd.DataFrame({
        'nyears': n_years,
        'nsamples': n_samples,
        'cumulative': total_return,
        'cagr': cagr,
        'mean': mean_return,
        'vol': vol,
        'skew': skew,
        'kurt': kurt,
        'max': returns.max(),
        'min': returns.min(),
        'sharpe': sharpe,
        'mdd': mdd,
    }).T

    return stats


def format_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """summary_stats() as display strings (stats table, stats DataFrame, report)."""
    formatted = stats.copy().astype(object)
    for row in ["cumulative", "cagr", "mean", "vol", "max", "min", "mdd"]:
        if row in formatted.index:
            formatted.loc[row] = stats.loc[row].apply(lambda x: f"{x:.1%}")
    for row in ["sharpe", "skew", "kurt"]:
        if row in formatted.index:
            formatted.loc[row] = stats.loc[row].apply(lambda x: f"{x:.2f}")
    if "nyears" in formatted.index:
        formatted.loc["nyears"] = stats.loc["nyears"].apply(lambda x: f"{x:.1f}")
    if "nsamples" in formatted.index:
        formatted.loc["nsamples"] = stats.loc["nsamples"].apply(lambda x: f"{int(x)}")
    return formatted


def kpi_lines(kpi: dict) -> list[str]:
    """One KPI card's lines, as shown on the dashboard and in the report."""
    return [
        f"CAGR: {kpi['cagr']:.1%}",
        f"Vol: {kpi['vol']:.1%}",
        f"Sharpe: {kpi['sharpe']:.2f}",
        f"MDD: {kpi['mdd']:.1%}",
        f"YTD: {kpi['ytd']:.1%}",
    ]


def rebase(navs: pd.DataFrame | pd.Series, start_date: str | None = None) -> pd.DataFrame:
    """
    Rebase NAV series to start at 100 from a given date.

    Args:
        navs: NAV DataFrame or Series
        start_date: Date to start rebasing from

    Returns:
        Rebased DataFrame
    """
    if isinstance(navs, pd.Series):
        navs = navs.to_frame()

    if start_date is not None:
        navs = navs[start_date:]

    rebased = navs / navs.iloc[0] * 100
    return rebased


def calculate_kpis(navs: pd.DataFrame, benchmark_col: str = 'BM') -> dict:
    """
    Calculate key performance indicators for dashboard display.

    Args:
        navs: NAV DataFrame with strategy and benchmark columns
        benchmark_col: Name of benchmark column

    Returns:
        Dictionary of KPIs
    """
    stats = summary_stats(navs)

    # Get strategy columns (excluding benchmark)
    strat_cols = [c for c in navs.columns if c != benchmark_col]

    kpis = {}
    for col in strat_cols:
        kpis[col] = {
            'cagr': stats.loc['cagr', col],
            'sharpe': stats.loc['sharpe', col],
            'vol': stats.loc['vol', col],
            'mdd': stats.loc['mdd', col],
            'ytd': _calculate_ytd_return(navs[col]),
            'mtd': _calculate_mtd_return(navs[col]),
        }

    if benchmark_col in navs.columns:
        kpis[benchmark_col] = {
            'cagr': stats.loc['cagr', benchmark_col],
            'sharpe': stats.loc['sharpe', benchmark_col],
            'vol': stats.loc['vol', benchmark_col],
            'mdd': stats.loc['mdd', benchmark_col],
            'ytd': _calculate_ytd_return(navs[benchmark_col]),
            'mtd': _calculate_mtd_return(navs[benchmark_col]),
        }

    return kpis


def _calculate_ytd_return(nav: pd.Series) -> float:
    """Calculate year-to-date return."""
    current_year = nav.index[-1].year
    ytd_start = nav[nav.index.year == current_year].iloc[0]
    return nav.iloc[-1] / ytd_start - 1


def _calculate_mtd_return(nav: pd.Series) -> float:
    """Calculate month-to-date return."""
    current_month = nav.index[-1].month
    current_year = nav.index[-1].year
    mtd_data = nav[(nav.index.year == current_year) & (nav.index.month == current_month)]
    if len(mtd_data) > 0:
        return nav.iloc[-1] / mtd_data.iloc[0] - 1
    return 0.0


def annual_returns(navs: pd.DataFrame) -> pd.DataFrame:
    """Calculate annual returns for each column. Last row is YTD if incomplete year.

    Args:
        navs: NAV DataFrame (daily index)

    Returns:
        DataFrame with years as index, columns matching navs columns.
        Values are decimal returns (e.g. 0.10 = 10%).
    """
    if isinstance(navs, pd.Series):
        navs = navs.to_frame()

    navs = navs.dropna()
    daily_ret = navs.pct_change().fillna(0)

    # Group by year and compound
    annual = daily_ret.groupby(daily_ret.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )
    annual.index.name = "Year"

    # Label last row as YTD if year is incomplete
    last_year = navs.index[-1].year
    last_day = navs.index[-1]
    year_end = pd.Timestamp(f"{last_year}-12-31")
    if last_day < year_end:
        annual = annual.rename(index={last_year: f"{last_year} YTD"})

    return annual


def monthly_returns(nav: pd.Series) -> pd.DataFrame:
    """Convert a single NAV series to a (year x month) matrix of returns.

    Args:
        nav: NAV Series (daily index)

    Returns:
        DataFrame with years as index, months 1-12 as columns.
        Values are decimal returns.
    """
    nav = nav.dropna()
    daily_ret = nav.pct_change().fillna(0)

    # Compound by (year, month)
    monthly = daily_ret.groupby([daily_ret.index.year, daily_ret.index.month]).apply(
        lambda x: (1 + x).prod() - 1
    )
    monthly.index.names = ["Year", "Month"]
    monthly = monthly.unstack(level="Month")

    # Ensure columns 1-12
    for m in range(1, 13):
        if m not in monthly.columns:
            monthly[m] = np.nan
    monthly = monthly[range(1, 13)]

    return monthly
