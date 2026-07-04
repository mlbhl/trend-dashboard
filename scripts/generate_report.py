"""디폴트 설정 분석을 headless로 실행해 HTML 리포트를 생성한다.

dash_components/callbacks.py의 run_analysis + update_charts 디폴트 경로
(Single 모드, Top-K, Equal Weight BM)를 Dash 없이 재현한다.
scripts/run_daily_report.sh가 cron으로 매일 호출한다.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BMonthBegin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.charts import (
    create_annual_returns_chart,
    create_drawdown_chart,
    create_holding_heatmap,
    create_monthly_returns_table,
    create_nav_chart,
    create_returns_table,
    create_signal_category_table,
)
from src.config import (
    DEFAULT_BACKTEST_START_DATE,
    DEFAULT_LONG_WEIGHT,
    DEFAULT_LONG_WINDOW,
    DEFAULT_MID_WEIGHT,
    DEFAULT_MID_WINDOW,
    DEFAULT_N_QUANTILES,
    DEFAULT_PRESET,
    DEFAULT_SHORT_WEIGHT,
    DEFAULT_SHORT_WINDOW,
    DEFAULT_START_DATE,
    DEFAULT_THRESH,
    DEFAULT_TOP_K,
    TICKER_PRESETS,
)
from src.data import load_price_data
from src.metrics import (
    annual_returns,
    calculate_kpis,
    monthly_returns,
    rebase,
    summary_stats,
)
from src.portfolio import compute_weight_top_k, run_top_k_backtest
from src.signals import generate_signal, get_signal_ranking


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


def _fig_html(fig, include_js: bool = False) -> str:
    """Plotly figure를 embed용 HTML 조각으로 변환. plotly.js는 첫 차트에만 CDN 로드."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if include_js else False,
        config={"displayModeBar": False},
    )


def _format_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """summary_stats 결과를 대시보드 stats-df와 동일하게 포맷 (callbacks.py와 동일 로직)."""
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


_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trend Dashboard Report — {report_date}</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", "Malgun Gothic", sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 24px; color: #212529; }}
h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
h2 {{ font-size: 1.2rem; margin-top: 2rem; border-bottom: 1px solid #dee2e6;
     padding-bottom: 4px; }}
.meta {{ color: #6c757d; margin: 2px 0; }}
.warning {{ color: #856404; background: #fff3cd; padding: 8px 12px;
           border-radius: 4px; margin: 12px 0; }}
.kpi-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }}
.kpi-card {{ border: 1px solid #dee2e6; border-radius: 6px; padding: 12px 16px;
            flex: 1 1 160px; }}
.kpi-card h3 {{ margin: 0 0 8px; font-size: 0.95rem; }}
.kpi-card p {{ margin: 2px 0; font-size: 0.85rem; }}
.table-scroll {{ overflow-x: auto; }}
table.data-table {{ border-collapse: collapse; font-size: 0.8rem; }}
table.data-table th, table.data-table td {{ border: 1px solid #dee2e6;
    padding: 4px 8px; text-align: center; }}
table.data-table th {{ background: #f8f9fa; }}
</style>
</head>
<body>
<h1>Trend Dashboard — Daily Report</h1>
<p class="meta">Generated: {generated_at}</p>
<p class="meta">{period_info}</p>
{warning_html}

<h2>KPI</h2>
<div class="kpi-row">{kpi_cards}</div>

<h2>Portfolio NAV</h2>
{nav_html}

<h2>Drawdown</h2>
{dd_html}

<h2>Current Signal</h2>
<p class="meta">Date: {data_end}</p>
<p class="meta">If the latest month is not complete yet, the signal is preliminary —
computed from the latest intra-month prices.</p>
{signal_html}

<h2>Raw Signal (Last 5 Months)</h2>
<p class="meta">Monthly signal rankings. If the last column's month is not complete yet,
it is a preliminary signal computed from the latest intra-month prices.</p>
<div class="table-scroll">{raw_signal_html}</div>

<h2>Annual Returns</h2>
{annual_html}

<h2>Monthly Returns — {strat_name}</h2>
{monthly_html}

<h2>Monthly Holdings</h2>
<p class="meta">Portfolio formed from each month-end signal and held for the following
month. Each date is the start of that holding month.</p>
{heatmap_html}

<h2>Stats</h2>
{stats_fig_html}
<div class="table-scroll">{stats_df_html}</div>
</body>
</html>
"""


def render_report_html(
    results: dict, missing: list[str], generated_at: pd.Timestamp
) -> str:
    """분석 결과를 웹페이지 오른편과 동일한 구성의 단일 HTML로 렌더링."""
    navs = results["navs"]
    signal = results["signal"]
    wgt = results["weights"]
    strat_name = results["strat_name"]
    bm_col = results["bm_col"]

    navs_rebased = rebase(navs)

    # KPI 카드
    kpis = calculate_kpis(navs, benchmark_col=bm_col)
    kpi_cards = "".join(
        f'<div class="kpi-card"><h3>{name}</h3>'
        f'<p>CAGR: {m["cagr"]:.1%}</p>'
        f'<p>Vol: {m["vol"]:.1%}</p>'
        f'<p>Sharpe: {m["sharpe"]:.2f}</p>'
        f'<p>MDD: {m["mdd"]:.1%}</p>'
        f'<p>YTD: {m["ytd"]:.1%}</p></div>'
        for name, m in kpis.items()
    )

    # 차트 (update_charts와 동일한 함수/인자)
    fig_nav = create_nav_chart(navs_rebased, title="Portfolio NAV (Rebased to 100)")
    fig_dd = create_drawdown_chart(navs_rebased, title="Drawdown (%)", height=350)
    fig_category = create_signal_category_table(
        get_signal_ranking(signal), n_quantiles=DEFAULT_N_QUANTILES, title=""
    )
    fig_annual = create_annual_returns_chart(annual_returns(navs), title="Annual Returns")
    fig_monthly = create_monthly_returns_table(
        monthly_returns(navs[strat_name]), title=""
    )
    recent_wgt = wgt.iloc[-24:]
    fig_heatmap = create_holding_heatmap(
        recent_wgt,
        title="",
        height=max(500, len(recent_wgt.columns) * 25),
    )
    stats = summary_stats(navs)
    fig_stats = create_returns_table(stats)

    # Raw signal 최근 5일 테이블
    signal_display = signal.iloc[-5:].T.copy()
    signal_display.columns = signal_display.columns.strftime("%Y-%m-%d")
    raw_signal_html = signal_display.to_html(
        classes="data-table", float_format=lambda x: f"{x:.0f}", na_rep=""
    )

    stats_df_html = _format_stats(stats).T.to_html(classes="data-table")

    warning_html = (
        f'<p class="warning">Data not found for: {", ".join(missing)}</p>'
        if missing
        else ""
    )

    return _TEMPLATE.format(
        report_date=results["display_end"],
        generated_at=generated_at.strftime("%Y-%m-%d %H:%M"),
        period_info=f"Backtest Period: {results['display_start']} ~ {results['display_end']}",
        warning_html=warning_html,
        kpi_cards=kpi_cards,
        nav_html=_fig_html(fig_nav, include_js=True),
        dd_html=_fig_html(fig_dd),
        data_end=results["data_end"],
        signal_html=_fig_html(fig_category),
        raw_signal_html=raw_signal_html,
        annual_html=_fig_html(fig_annual),
        strat_name=strat_name,
        monthly_html=_fig_html(fig_monthly),
        heatmap_html=_fig_html(fig_heatmap),
        stats_fig_html=_fig_html(fig_stats),
        stats_df_html=stats_df_html,
    )


def main() -> int:
    tickers = TICKER_PRESETS[DEFAULT_PRESET]
    print(f"가격 데이터 다운로드: {len(tickers)}개 티커, start={DEFAULT_START_DATE}")
    dataset, missing = load_price_data(tickers, start_date=DEFAULT_START_DATE)

    if len(dataset.columns) < 2:
        print(f"ERROR: 유효 티커 부족 (missing: {missing})", file=sys.stderr)
        return 1
    if missing:
        print(f"WARNING: 데이터 없음 — {', '.join(missing)}")

    results = run_default_analysis(dataset)
    now = datetime.now()
    html = render_report_html(results, missing=missing, generated_at=pd.Timestamp(now))

    outdir = PROJECT_ROOT / "reports"
    outdir.mkdir(exist_ok=True)
    out = outdir / f"report_{now:%Y-%m-%d}.html"
    out.write_text(html, encoding="utf-8")
    print(f"리포트 생성 완료: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
