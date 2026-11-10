"""Shared synthetic price fixtures (network-free, deterministic)."""
import numpy as np
import pandas as pd
import pytest

START = "2015-01-01"
PERIODS = 1300  # ~5 years of business days (2015-01 .. 2019-12)


def make_prices(
    n_tickers: int = 12,
    seed: int = 0,
    late_tickers: int = 0,
    late_start: str = "2016-01-01",
    start: str = START,
) -> pd.DataFrame:
    """Random-walk daily prices; the last `late_tickers` columns are NaN before `late_start`."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=PERIODS)
    drift = np.linspace(-0.0003, 0.0006, n_tickers)
    rets = rng.normal(drift, 0.01, size=(PERIODS, n_tickers))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rets, axis=0)),
        index=idx,
        columns=[f"T{i:02d}" for i in range(n_tickers)],
    )
    if late_tickers:
        prices.loc[prices.index < late_start, prices.columns[-late_tickers:]] = np.nan
    return prices


@pytest.fixture
def prices() -> pd.DataFrame:
    """12 tickers, full history."""
    return make_prices()


@pytest.fixture
def prices_late() -> pd.DataFrame:
    """12 tickers, last 3 start in 2016 (like ETFs listed later than the rest)."""
    return make_prices(late_tickers=3)
