"""scripts/generate_report.py 테스트 — 네트워크 없이 합성 가격 데이터 사용."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_report import render_report_html, run_default_analysis


def make_dataset(n_tickers: int = 12, start: str = "2013-01-01", periods: int = 1300) -> pd.DataFrame:
    """5년치(2013~2017) 합성 일간 가격. thresh=10을 넘도록 12종목."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(start, periods=periods)
    data = {
        f"T{i:02d}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))
        for i in range(n_tickers)
    }
    return pd.DataFrame(data, index=idx)


def test_run_default_analysis_shapes():
    dataset = make_dataset()
    results = run_default_analysis(dataset)

    navs = results["navs"]
    assert list(navs.columns) == ["Top-5", "BM (Equal Weight)"]
    # 백테스트 시작일(2015-01-01) 이전은 잘려야 함
    assert navs.index[0] >= pd.Timestamp("2015-01-01")
    assert navs.notna().all().all()

    wgt = results["weights"]
    # 보유 시점 가중치는 합이 1 (equal weight top-5)
    row_sums = wgt.sum(axis=1)
    assert np.allclose(row_sums[row_sums > 0], 1.0)
    # 가중치 index는 표시 구간(display_end)을 넘지 않음
    assert wgt.index[-1] <= navs.index[-1]

    assert results["strat_name"] == "Top-5"
    assert results["bm_col"] == "BM (Equal Weight)"
    assert results["display_start"] == navs.index[0].strftime("%Y-%m-%d")
    assert results["display_end"] == navs.index[-1].strftime("%Y-%m-%d")
    assert results["data_end"] == dataset.index[-1].strftime("%Y-%m-%d")


def test_render_report_html_contains_all_sections():
    dataset = make_dataset()
    results = run_default_analysis(dataset)
    html = render_report_html(
        results, missing=["FAKE"], generated_at=pd.Timestamp("2026-07-03 08:10")
    )

    for expected in [
        "<!DOCTYPE html>",
        "Generated: 2026-07-03 08:10",
        "Backtest Period:",
        "Portfolio NAV",
        "Drawdown",
        "Current Signal",
        "Raw Signal (Last 5 Months)",
        "preliminary signal",
        "Annual Returns",
        "Monthly Returns",
        "Monthly Holdings",
        "held for the following",
        "Stats",
        "cdn.plot.ly",  # plotly.js CDN 로드 확인
        "Data not found for: FAKE",
    ]:
        assert expected in html, f"missing section: {expected}"

    # plotly.js는 정확히 한 번만 CDN 로드 (embed 회귀 시 파일이 ~4MB로 불어남)
    assert html.count("cdn.plot.ly") == 1
    assert len(html) < 2_000_000
