"""Data loading and preprocessing functions."""

import warnings
import pandas as pd
import yfinance as yf

warnings.filterwarnings(action='ignore')


def load_price_data(
    tickers: list[str],
    start_date: str = '2000-01-01',
    end_date: str | None = None,
    proxy: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load price data from Yahoo Finance.

    Args:
        tickers: List of ticker symbols
        start_date: Start date for data download
        end_date: End date (None for latest available)
        proxy: Proxy URL (e.g., 'http://proxy.example.com:8080')

    Returns:
        Tuple of (DataFrame with adjusted close prices, list of missing tickers)
    """
    # Set proxy if provided
    if proxy:
        yf.set_config(proxy=proxy)

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    # Handle different yfinance column formats
    if isinstance(data.columns, pd.MultiIndex):
        raw = data['Close']
    else:
        raw = data['Close'] if 'Close' in data.columns else data

    # Handle single ticker case
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])

    # Check which tickers have actual data (not all NaN)
    available_tickers = []
    missing_tickers = []

    for t in tickers:
        if t in raw.columns and raw[t].notna().any():
            available_tickers.append(t)
        else:
            missing_tickers.append(t)

    dataset = raw[available_tickers].resample('B').last().ffill()

    return dataset, missing_tickers


def get_latest_prices(dataset: pd.DataFrame) -> pd.Series:
    """Get the most recent prices for each ticker."""
    return dataset.iloc[-1]


def get_price_returns(dataset: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """Calculate returns over a given period."""
    return dataset.pct_change(period)
