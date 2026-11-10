"""Data loading and preprocessing functions."""

import pandas as pd
import yfinance as yf


def load_price_data(
    tickers: list[str],
    start_date: str = '2000-01-01',
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load price data from Yahoo Finance, up to the latest close.

    Args:
        tickers: List of ticker symbols
        start_date: Start date for data download

    Returns:
        Tuple of (DataFrame with adjusted close prices, list of missing tickers)
    """
    data = yf.download(
        tickers,
        start=start_date,
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
