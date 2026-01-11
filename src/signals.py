"""Signal generation functions for momentum strategies."""

import numpy as np
import pandas as pd


def generate_signal(
    price: pd.DataFrame,
    short_window: int = 1,
    mid_window: int = 3,
    long_window: int = 12,
    short_wgt: float = 1/3,
    mid_wgt: float = 1/3,
    long_wgt: float = 1/3,
) -> pd.DataFrame:
    """
    Generate momentum-based ranking signal.

    Args:
        price: Daily price DataFrame
        short_window: Short-term momentum lookback (months)
        mid_window: Mid-term momentum lookback (months)
        long_window: Long-term momentum lookback (months)
        short_wgt: Weight for short-term momentum
        mid_wgt: Weight for mid-term momentum
        long_wgt: Weight for long-term momentum

    Returns:
        DataFrame with cross-sectional rankings (1 = best)
    """
    price_m = price.resample('BM').last()

    # Calculate momentum for each window
    mom_s = (price_m / price_m.shift(short_window)).rank(axis=1)
    mom_m = (price_m / price_m.shift(mid_window)).rank(axis=1)
    mom_l = (price_m / price_m.shift(long_window)).rank(axis=1)

    # Weighted combination and re-rank (ascending=False means rank 1 = highest score)
    sig = (mom_s * short_wgt + mom_m * mid_wgt + mom_l * long_wgt).rank(axis=1, ascending=False)
    sig = sig.dropna(thresh=10)

    return sig


def calc_vol(
    price: pd.DataFrame,
    ew: bool = False,
    com: int = 60,
    window: int = 22,
) -> pd.DataFrame:
    """
    Calculate annualized volatility.

    Args:
        price: Daily price DataFrame
        ew: Use exponentially weighted if True, rolling if False
        com: Center of mass for EW calculation
        window: Rolling window for simple rolling calculation

    Returns:
        DataFrame with annualized volatility
    """
    ret = price.pct_change()

    if ew:
        vol = ret.ewm(com=com, min_periods=com * 2).std() * np.sqrt(260)
    else:
        vol = ret.rolling(window).std() * np.sqrt(260)

    return vol


def get_signal_ranking(signal: pd.DataFrame) -> pd.Series:
    """Get the most recent signal ranking, sorted by rank."""
    latest = signal.iloc[-1].dropna().sort_values()
    return latest
