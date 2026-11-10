"""Performance metrics on hand-checkable NAV paths."""
import numpy as np
import pandas as pd
import pytest

from src.config import PERIODS_PER_YEAR
from src.metrics import annual_returns, calculate_kpis, monthly_returns, rebase, sharpe_from_nav, summary_stats


def test_summary_stats_known_path():
    nav = pd.Series([100.0, 110.0, 99.0, 120.0], index=pd.bdate_range("2020-01-01", periods=4))
    s = summary_stats(nav).iloc[:, 0]
    rets = np.array([0.1, -0.1, 120 / 99 - 1])

    assert s["nsamples"] == 3
    assert s["cumulative"] == pytest.approx(0.2)
    assert s["cagr"] == pytest.approx(1.2 ** (PERIODS_PER_YEAR / 3) - 1)
    assert s["mdd"] == pytest.approx(99 / 110 - 1)
    assert s["mean"] == pytest.approx(rets.mean() * PERIODS_PER_YEAR)
    assert s["vol"] == pytest.approx(rets.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    assert s["sharpe"] == pytest.approx(s["mean"] / s["vol"])
    assert s["max"] == pytest.approx(rets.max()) and s["min"] == pytest.approx(rets.min())


def test_cagr_matches_the_data_frequency():
    """Prices are on every weekday (~261 a year): 10% a year for 10 years must read as 10% CAGR."""
    assert len(pd.bdate_range("2021-01-01", "2021-12-31")) == PERIODS_PER_YEAR
    n = PERIODS_PER_YEAR * 10
    nav = pd.Series(100 * 1.1 ** (np.arange(n + 1) / PERIODS_PER_YEAR), index=pd.bdate_range("2015-01-01", periods=n + 1))
    s = summary_stats(nav).iloc[:, 0]
    assert s["nyears"] == pytest.approx(10.0)
    assert s["cagr"] == pytest.approx(0.10, rel=1e-12)


def test_sharpe_from_nav_matches_summary_stats():
    base = _growth_nav("2020-01-01", "2021-06-30")
    nav = base * (1 + 0.01 * np.sin(np.arange(len(base))))
    assert sharpe_from_nav(nav) == pytest.approx(summary_stats(nav).loc["sharpe"].iloc[0], rel=1e-12)
    assert np.isnan(sharpe_from_nav(None)) and np.isnan(sharpe_from_nav(nav.iloc[:1]))


def _growth_nav(start: str, end: str, daily: float = 0.0005) -> pd.Series:
    idx = pd.bdate_range(start, end)
    return pd.Series(100 * (1 + daily) ** np.arange(len(idx)), index=idx)


def test_annual_returns_compound_and_label_ytd():
    nav = _growth_nav("2020-01-01", "2021-06-30")
    ann = annual_returns(nav).iloc[:, 0]

    end20 = nav[:"2020-12-31"].iloc[-1]
    assert list(ann.index) == [2020, "2021 YTD"]
    assert ann[2020] == pytest.approx(end20 / nav.iloc[0] - 1)
    assert ann["2021 YTD"] == pytest.approx(nav.iloc[-1] / end20 - 1)


def test_monthly_returns_grid():
    nav = _growth_nav("2020-01-01", "2020-12-31")
    m = monthly_returns(nav)

    assert list(m.columns) == list(range(1, 13)) and list(m.index) == [2020]
    feb_end, mar_end = nav[:"2020-02-29"].iloc[-1], nav[:"2020-03-31"].iloc[-1]
    assert m.loc[2020, 3] == pytest.approx(mar_end / feb_end - 1)


def test_kpis_ytd_mtd():
    nav = _growth_nav("2020-01-01", "2021-03-15")
    navs = pd.DataFrame({"Strat": nav, "BM": nav * 0.5})
    k = calculate_kpis(navs, "BM")

    assert set(k) == {"Strat", "BM"}
    ytd_base = nav[:"2020-12-31"].iloc[-1]   # previous year-end close
    mtd_base = nav[:"2021-02-28"].iloc[-1]   # previous month-end close
    assert k["Strat"]["ytd"] == pytest.approx(nav.iloc[-1] / ytd_base - 1)
    assert k["Strat"]["mtd"] == pytest.approx(nav.iloc[-1] / mtd_base - 1)
    assert k["BM"]["cagr"] == pytest.approx(k["Strat"]["cagr"])  # scaling a NAV changes no return metric


def test_kpi_ytd_and_mtd_match_the_annual_and_monthly_tables():
    base = _growth_nav("2019-06-01", "2021-03-15")
    nav = base * (1 + 0.02 * np.sin(np.arange(len(base)) / 7))
    k = calculate_kpis(pd.DataFrame({"S": nav}), "BM")["S"]
    assert k["ytd"] == pytest.approx(annual_returns(nav).iloc[-1, 0], rel=1e-12)
    assert k["mtd"] == pytest.approx(monthly_returns(nav).loc[2021, 3], rel=1e-12)


def test_ytd_falls_back_to_first_value_when_series_starts_this_year():
    nav = _growth_nav("2021-02-10", "2021-03-15")
    k = calculate_kpis(pd.DataFrame({"S": nav}), "BM")["S"]
    assert k["ytd"] == pytest.approx(nav.iloc[-1] / nav.iloc[0] - 1)
    assert k["ytd"] == pytest.approx(annual_returns(nav).iloc[-1, 0], rel=1e-12)


def test_rebase_starts_at_100():
    nav = _growth_nav("2020-01-01", "2020-06-30")
    r = rebase(nav, "2020-03-02")
    assert r.index[0] == pd.Timestamp("2020-03-02") and r.iloc[0, 0] == pytest.approx(100.0)
