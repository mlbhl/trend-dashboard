"""Performance metrics on hand-checkable NAV paths."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import annual_returns, calculate_kpis, monthly_returns, rebase, summary_stats


def test_summary_stats_known_path():
    nav = pd.Series([100.0, 110.0, 99.0, 120.0], index=pd.bdate_range("2020-01-01", periods=4))
    s = summary_stats(nav).iloc[:, 0]
    rets = np.array([0.1, -0.1, 120 / 99 - 1])

    assert s["nsamples"] == 3
    assert s["cumulative"] == pytest.approx(0.2)
    assert s["cagr"] == pytest.approx(1.2 ** (252 / 3) - 1)
    assert s["mdd"] == pytest.approx(99 / 110 - 1)
    assert s["mean"] == pytest.approx(rets.mean() * 252)
    assert s["vol"] == pytest.approx(rets.std(ddof=1) * np.sqrt(252))
    assert s["sharpe"] == pytest.approx(s["mean"] / s["vol"])
    assert s["max"] == pytest.approx(rets.max()) and s["min"] == pytest.approx(rets.min())


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
    ytd_base = nav["2021"].iloc[0]
    mtd_base = nav["2021-03"].iloc[0]
    assert k["Strat"]["ytd"] == pytest.approx(nav.iloc[-1] / ytd_base - 1)
    assert k["Strat"]["mtd"] == pytest.approx(nav.iloc[-1] / mtd_base - 1)
    assert k["BM"]["cagr"] == pytest.approx(k["Strat"]["cagr"])  # scaling a NAV changes no return metric


def test_rebase_starts_at_100():
    nav = _growth_nav("2020-01-01", "2020-06-30")
    r = rebase(nav, "2020-03-02")
    assert r.index[0] == pd.Timestamp("2020-03-02") and r.iloc[0, 0] == pytest.approx(100.0)
