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

    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )['Close']

    # Handle single ticker case
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])

    # Reorder columns to match input ticker order
    available_tickers = [t for t in tickers if t in raw.columns]
    missing_tickers = [t for t in tickers if t not in raw.columns]
    dataset = raw[available_tickers].resample('B').last().ffill()

    return dataset, missing_tickers


def get_latest_prices(dataset: pd.DataFrame) -> pd.Series:
    """Get the most recent prices for each ticker."""
    return dataset.iloc[-1]


def get_price_returns(dataset: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """Calculate returns over a given period."""
    return dataset.pct_change(period)
