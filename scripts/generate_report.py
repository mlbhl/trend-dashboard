"""대시보드 디폴트 설정의 Run Analysis 결과를 headless로 HTML 리포트로 만든다 (수동 실행).

계산은 대시보드와 같은 src/analysis.run_analysis()를, 표 포맷도 같은 헬퍼를 쓴다.
차트만 정적 이미지(matplotlib, report_charts.py)로 그린다.

    python scripts/generate_report.py   →  reports/report_<날짜>.html
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from report_charts import (
    annual_returns_img,
    drawdown_chart_img,
    holding_heatmap_img,
    monthly_returns_html,
    nav_chart_img,
    signal_category_html,
)
from src.analysis import AnalysisError, AnalysisResult, run_analysis
from src.config import DEFAULT_PRESET, DEFAULT_START_DATE, TICKER_PRESETS
from src.metrics import (
    annual_returns,
    calculate_kpis,
    format_stats,
    kpi_lines,
    monthly_returns,
    rebase,
    summary_stats,
)
from src.signals import get_signal_ranking, signal_history


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
table.signal-table {{ margin-top: 8px; }}
table.signal-table th {{ font-weight: bold; }}
table.monthly-table th {{ background: #f8f9fa; }}
img {{ display: block; margin: 8px 0; }}
</style>
</head>
<body>
<h1>Theme Momentum Dashboard — Daily Report</h1>
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
<div class="table-scroll">{signal_html}</div>

<h2>Signal History</h2>
<p class="meta">Monthly signal rankings. If the last column's month is not complete yet,
it is a preliminary signal computed from the latest intra-month prices.</p>
<div class="table-scroll">{raw_signal_html}</div>

<h2>Annual Returns</h2>
{annual_html}

<h2>Monthly Returns — {strat_name}</h2>
<div class="table-scroll">{monthly_html}</div>

<h2>Monthly Holdings</h2>
<p class="meta">Portfolio formed from each month-end signal and held for the following
month. Each date is the rebalance date (month-end signal + implementation lag).</p>
{heatmap_html}

<h2>Statistics</h2>
<div class="table-scroll">{stats_df_html}</div>
</body>
</html>
"""


def render_report_html(result: AnalysisResult, generated_at: pd.Timestamp) -> str:
    """분석 결과를 웹페이지 오른편과 동일한 구성의 단일 HTML로 렌더링."""
    navs = result.navs
    signal = result.signal
    params = result.params
    strat_name = result.strat_name
    wgt = result.weights[strat_name]

    navs_rebased = rebase(navs)

    # KPI 카드
    kpis = calculate_kpis(navs, benchmark_col=params["bm_col_name"])
    kpi_cards = "".join(
        f'<div class="kpi-card"><h3>{name}</h3>'
        + "".join(f"<p>{line}</p>" for line in kpi_lines(m))
        + "</div>"
        for name, m in kpis.items()
    )

    # 차트 (matplotlib PNG) / 표 (HTML) — 계산 입력은 update_charts와 동일
    nav_html = nav_chart_img(navs_rebased, title="Portfolio NAV (Rebased to 100)")
    dd_html = drawdown_chart_img(navs_rebased, title="Drawdown (%)")
    signal_html = signal_category_html(get_signal_ranking(signal), n_quantiles=params["n_quantiles"])
    annual_html = annual_returns_img(annual_returns(navs), title="Annual Returns")
    monthly_html = monthly_returns_html(monthly_returns(navs[strat_name]))
    heatmap_html = holding_heatmap_img(wgt.iloc[-24:], title="")
    raw_signal_html = signal_history(signal, params["data_end"]).to_html(
        classes="data-table", float_format=lambda x: f"{x:.0f}", na_rep=""
    )
    stats_df_html = format_stats(summary_stats(navs)).T.to_html(classes="data-table")

    warning_html = "".join(f'<p class="warning">{msg}</p>' for msg in result.warnings)

    return _TEMPLATE.format(
        report_date=params["display_end"],
        generated_at=generated_at.strftime("%Y-%m-%d %H:%M"),
        period_info=f"Backtest Period: {params['display_start']} ~ {params['display_end']}",
        warning_html=warning_html,
        kpi_cards=kpi_cards,
        nav_html=nav_html,
        dd_html=dd_html,
        data_end=params["data_end"],
        signal_html=signal_html,
        raw_signal_html=raw_signal_html,
        annual_html=annual_html,
        strat_name=strat_name,
        monthly_html=monthly_html,
        heatmap_html=heatmap_html,
        stats_df_html=stats_df_html,
    )


def main() -> int:
    tickers = TICKER_PRESETS[DEFAULT_PRESET]
    print(f"가격 데이터 다운로드: {len(tickers)}개 티커, start={DEFAULT_START_DATE}")
    try:
        result = run_analysis(tickers)  # 사이드바 디폴트 설정
    except AnalysisError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    for msg in result.warnings:
        print(f"WARNING: {msg}")

    now = datetime.now()
    html = render_report_html(result, generated_at=pd.Timestamp(now))

    outdir = PROJECT_ROOT / "reports"
    outdir.mkdir(exist_ok=True)
    out = outdir / f"report_{now:%Y-%m-%d}.html"
    out.write_text(html, encoding="utf-8")
    print(f"리포트 생성 완료: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
