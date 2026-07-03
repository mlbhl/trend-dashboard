"""디폴트 설정 분석을 headless로 실행해 HTML 리포트를 생성한다.

dash_components/callbacks.py의 run_analysis + update_charts 디폴트 경로
(Single 모드, Top-K, Equal Weight BM)를 Dash 없이 재현한다.
scripts/run_daily_report.sh가 cron으로 매일 호출한다.
"""

import sys
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BMonthBegin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_BACKTEST_START_DATE,
    DEFAULT_LONG_WEIGHT,
    DEFAULT_LONG_WINDOW,
    DEFAULT_MID_WEIGHT,
    DEFAULT_MID_WINDOW,
    DEFAULT_PRESET,
    DEFAULT_SHORT_WEIGHT,
    DEFAULT_SHORT_WINDOW,
    DEFAULT_START_DATE,
    DEFAULT_THRESH,
    DEFAULT_TOP_K,
    TICKER_PRESETS,
)
from src.portfolio import compute_weight_top_k, run_top_k_backtest
from src.signals import generate_signal


def run_default_analysis(dataset: pd.DataFrame) -> dict:
    """사이드바 디폴트(Single 모드, Top-K, Equal Weight BM)로 분석 실행."""
    # 윈도우 가중치 정규화 (run_analysis와 동일)
    total = DEFAULT_SHORT_WEIGHT + DEFAULT_MID_WEIGHT + DEFAULT_LONG_WEIGHT
    short_wgt = DEFAULT_SHORT_WEIGHT / total
    mid_wgt = DEFAULT_MID_WEIGHT / total
    long_wgt = DEFAULT_LONG_WEIGHT / total

    signal = generate_signal(
        dataset,
        short_window=DEFAULT_SHORT_WINDOW,
        mid_window=DEFAULT_MID_WINDOW,
        long_window=DEFAULT_LONG_WINDOW,
        short_wgt=short_wgt,
        mid_wgt=mid_wgt,
        long_wgt=long_wgt,
        thresh=DEFAULT_THRESH,
    )

    strat_nav, bm_nav, _ = run_top_k_backtest(
        dataset, signal, top_k=DEFAULT_TOP_K, weight_method="equal", tcost=0.0
    )
    strat_name = f"Top-{DEFAULT_TOP_K}"
    bm_col = "BM (Equal Weight)"
    navs = pd.DataFrame({strat_name: strat_nav, bm_col: bm_nav})

    # Holdings heatmap용 가중치 — BMonthBegin(1) shift가 실제 적용 시점
    wgt, _ = compute_weight_top_k(
        dataset, signal, top_k=DEFAULT_TOP_K, weight_method="equal"
    )
    wgt.index = wgt.index + BMonthBegin(1)

    # 백테스트 시작일 이전 구간 절단 (종료일은 최신까지)
    backtest_start = pd.to_datetime(DEFAULT_BACKTEST_START_DATE)
    after_start = navs.index[navs.index >= backtest_start]
    display_start = after_start[0] if len(after_start) > 0 else navs.index[0]
    navs_display = navs.loc[display_start:].copy()
    display_end = navs_display.index[-1]

    wgt = wgt.sort_index().loc[:display_end]

    return {
        "navs": navs_display,
        "signal": signal,
        "weights": wgt,
        "strat_name": strat_name,
        "bm_col": bm_col,
        "display_start": display_start.strftime("%Y-%m-%d"),
        "display_end": display_end.strftime("%Y-%m-%d"),
        "data_end": dataset.index[-1].strftime("%Y-%m-%d"),
    }
