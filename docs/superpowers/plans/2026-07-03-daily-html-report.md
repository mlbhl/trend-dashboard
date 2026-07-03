# Daily HTML Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 매일 KST 8:10/12:10/16:10에 cron으로 사이드바 디폴트 분석을 headless 실행해 self-contained HTML 리포트를 `reports/`에 저장하고 git에 커밋/푸시한다.

**Architecture:** `scripts/generate_report.py`가 `dash_components/callbacks.py`의 `run_analysis`+`update_charts` 디폴트 경로를 `src/` 함수 직접 호출로 재현한다. 순수 함수 2개(`run_default_analysis`, `render_report_html`)는 합성 데이터로 테스트하고, `main()`만 네트워크(yfinance)를 탄다. cron 래퍼 `scripts/run_daily_report.sh`는 ETF Scout의 `run_daily.sh` 패턴(멱등 가드 + 로그 + git push)을 따른다.

**Tech Stack:** Python 3.10+ (miniconda base), pandas, plotly, yfinance, pytest 9, bash, cron

**Spec:** `docs/superpowers/specs/2026-07-03-daily-html-report-design.md`

## Global Constraints

- Python 인터프리터는 항상 `/home/byoun/miniconda3/bin/python3` (프로젝트 venv 없음, base env에 의존성 확인됨)
- 커밋 메시지는 단일 라인, 간단하게 (예: "Add daily report generator"). Co-Authored-By나 Claude attribution 금지 (사용자 글로벌 설정)
- plotly.js는 CDN 로드 (`include_plotlyjs="cdn"`) — 리포트가 git에 커밋되므로 파일 크기를 수백 KB로 유지
- 리포트 파라미터는 사이드바 디폴트와 동일해야 함: Alpha 프리셋(20종목), Top-K K=5, 윈도우 1/3/11, 가중치 0.10/0.10/0.80, thresh=10, weight_method="equal", tcost=0.0, BM=Equal Weight, 데이터 시작 2000-01-01, 백테스트 시작 2015-01-01, 백테스트 종료 없음(최신까지)
- 테스트는 네트워크를 타지 않는다 (합성 가격 데이터 사용). `main()`만 yfinance 호출
- 테스트 실행: `/home/byoun/miniconda3/bin/python3 -m pytest tests/ -v`

---

### Task 1: 분석 코어 — `run_default_analysis()`

**Files:**
- Create: `scripts/generate_report.py`
- Create: `tests/test_generate_report.py`

**Interfaces:**
- Consumes: `src.signals.generate_signal`, `src.portfolio.run_top_k_backtest`, `src.portfolio.compute_weight_top_k`, `src.config`의 DEFAULT_* 상수
- Produces: `run_default_analysis(dataset: pd.DataFrame) -> dict` — keys: `navs` (DataFrame, columns `["Top-5", "BM (Equal Weight)"]`), `signal` (DataFrame), `weights` (DataFrame, BMonthBegin(1) shift + display_end 절단), `strat_name` (str), `bm_col` (str), `display_start`/`display_end`/`data_end` (str, `"%Y-%m-%d"`)

- [ ] **Step 1: Write the failing test**

`tests/test_generate_report.py` 생성:

```python
"""scripts/generate_report.py 테스트 — 네트워크 없이 합성 가격 데이터 사용."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_report import run_default_analysis


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/byoun/miniconda3/bin/python3 -m pytest tests/test_generate_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.generate_report'` (또는 ImportError)

- [ ] **Step 3: Write minimal implementation**

`scripts/generate_report.py` 생성:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/byoun/miniconda3/bin/python3 -m pytest tests/test_generate_report.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_report.py tests/test_generate_report.py
git commit -m "Add headless default analysis for daily report"
```

---

### Task 2: HTML 렌더링 — `render_report_html()`

**Files:**
- Modify: `scripts/generate_report.py` (함수 추가)
- Modify: `tests/test_generate_report.py` (테스트 추가)

**Interfaces:**
- Consumes: Task 1의 `run_default_analysis()` 반환 dict, `src.charts`의 차트 함수들, `src.metrics`의 `calculate_kpis`/`rebase`/`annual_returns`/`monthly_returns`/`summary_stats`, `src.signals.get_signal_ranking`, `src.config`의 `DEFAULT_N_QUANTILES`/`TICKER_DESCRIPTIONS`
- Produces: `render_report_html(results: dict, missing: list[str], generated_at: pd.Timestamp) -> str` — 완전한 HTML 문서 문자열

- [ ] **Step 1: Write the failing test**

`tests/test_generate_report.py`에 추가 (import 라인도 수정):

```python
from scripts.generate_report import render_report_html, run_default_analysis
```

```python
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
        "Raw Signal",
        "Annual Returns",
        "Monthly Returns",
        "Monthly Holdings",
        "Stats",
        "cdn.plot.ly",  # plotly.js CDN 로드 확인
        "Data not found for: FAKE",
    ]:
        assert expected in html, f"missing section: {expected}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/byoun/miniconda3/bin/python3 -m pytest tests/test_generate_report.py -v`
Expected: FAIL — `ImportError: cannot import name 'render_report_html'`

- [ ] **Step 3: Write implementation**

`scripts/generate_report.py`에 추가. import 블록에 다음을 추가:

```python
from src.charts import (
    create_annual_returns_chart,
    create_drawdown_chart,
    create_holding_heatmap,
    create_monthly_returns_table,
    create_nav_chart,
    create_returns_table,
    create_signal_category_table,
)
from src.config import DEFAULT_N_QUANTILES  # 기존 config import 블록에 추가
from src.metrics import (
    annual_returns,
    calculate_kpis,
    monthly_returns,
    rebase,
    summary_stats,
)
from src.signals import get_signal_ranking  # 기존 signals import에 추가
```

함수 추가:

```python
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
{signal_html}

<h2>Raw Signal (Last 5 Days)</h2>
<div class="table-scroll">{raw_signal_html}</div>

<h2>Annual Returns</h2>
{annual_html}

<h2>Monthly Returns — {strat_name}</h2>
{monthly_html}

<h2>Monthly Holdings</h2>
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/byoun/miniconda3/bin/python3 -m pytest tests/test_generate_report.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_report.py tests/test_generate_report.py
git commit -m "Add HTML report rendering"
```

---

### Task 3: `main()` — 데이터 다운로드 및 파일 출력

**Files:**
- Modify: `scripts/generate_report.py` (main 추가)

**Interfaces:**
- Consumes: Task 1의 `run_default_analysis`, Task 2의 `render_report_html`, `src.data.load_price_data`
- Produces: CLI 실행 시 `reports/report_YYYY-MM-DD.html` 생성, 성공 시 exit code 0 / 실패 시 1

- [ ] **Step 1: Write implementation**

`scripts/generate_report.py` 상단 import에 추가:

```python
from datetime import datetime

from src.data import load_price_data
```

파일 끝에 추가:

```python
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
```

- [ ] **Step 2: Run E2E (실제 네트워크 사용)**

Run: `/home/byoun/miniconda3/bin/python3 scripts/generate_report.py`
Expected: "리포트 생성 완료: .../reports/report_<오늘날짜>.html" 출력, exit code 0

- [ ] **Step 3: Verify output file**

Run: `ls -la reports/ && /home/byoun/miniconda3/bin/python3 -c "html = open('reports/report_$(date +%F).html').read(); assert 'Portfolio NAV' in html and 'cdn.plot.ly' in html; print(f'OK, {len(html)//1024}KB')"`
Expected: 파일 존재, "OK" 출력, 크기 수백 KB 수준 (4MB면 plotly embed 실수)

가능하면 브라우저에서 열어 10개 섹션이 모두 렌더링되는지, 대시보드 Run Analysis 결과와 KPI 수치가 일치하는지 확인.

- [ ] **Step 4: Run full test suite (regression)**

Run: `/home/byoun/miniconda3/bin/python3 -m pytest tests/ -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit (리포트 파일 포함)**

```bash
git add scripts/generate_report.py reports/
git commit -m "Add daily report generator entry point"
```

---

### Task 4: cron 래퍼 — `scripts/run_daily_report.sh`

**Files:**
- Create: `scripts/run_daily_report.sh`
- Modify: `.gitignore` (마지막에 추가)

**Interfaces:**
- Consumes: Task 3의 `scripts/generate_report.py` CLI (exit code 0 = 성공, `reports/report_YYYY-MM-DD.html` 생성)
- Produces: cron에서 호출 가능한 실행 스크립트 — 멱등 가드, `reports/latest.html` 복사, git 커밋/푸시, `reports/cron.log` 로깅

- [ ] **Step 1: Write the script**

`scripts/run_daily_report.sh` 생성 (ETF Scout `run_daily.sh` 패턴):

```bash
#!/usr/bin/env bash
# Trend Dashboard 일일 리포트 실행 래퍼 (cron용)
#
# 하루 3번(8:10/12:10/16:10 KST) 시도하되, 멱등 가드로 실제 실행은 1회.
# 8:10에 PC가 꺼져 있었으면 다음 시각에 따라잡는다.
set -uo pipefail

# cron은 최소 환경으로 실행되므로 HOME/PATH를 명시한다 (git push에 gh 크리덴셜 필요).
export HOME="/home/byoun"
export PATH="/home/byoun/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

PROJECT_DIR="/home/byoun/projects/trend-dashboard"
PYTHON="/home/byoun/miniconda3/bin/python3"
cd "$PROJECT_DIR" || exit 1
mkdir -p reports

{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') 실행 시작 ====="
  today=$(date +%F)
  report="reports/report_${today}.html"

  # 멱등 가드: 오늘치 리포트가 이미 있으면 스킵
  if [ -f "$report" ]; then
    echo "오늘(${today}) 이미 생성됨 — 스킵"
  else
    "$PYTHON" scripts/generate_report.py
    code=$?
    echo "python 종료 코드: ${code}"

    if [ "$code" -eq 0 ] && [ -f "$report" ]; then
      cp "$report" reports/latest.html
      git add reports
      if git diff --staged --quiet; then
        echo "변경 없음 — 커밋 생략"
      else
        git commit -m "Update daily report"
        git push || echo "git push 실패 — 리포트 파일은 보존됨, 다음 커밋 때 함께 푸시"
      fi
    else
      echo "리포트 생성 실패 — 다음 시각에 재시도"
    fi
  fi
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') 실행 종료 ====="
} >> reports/cron.log 2>&1
```

- [ ] **Step 2: Make executable, update .gitignore**

```bash
chmod +x scripts/run_daily_report.sh
```

`.gitignore` 마지막에 추가:

```
# Daily report log
reports/cron.log
```

- [ ] **Step 3: Test full path (오늘 리포트 삭제 후 실행)**

Task 3에서 만든 오늘치 리포트는 재생성 가능하므로 삭제하고 전체 경로(생성→latest 복사→커밋→푸시)를 검증한다. 사전에 git 상태가 깨끗한지 확인:

```bash
git status --short   # 깨끗해야 함 (아니면 먼저 커밋)
rm reports/report_$(date +%F).html
./scripts/run_daily_report.sh
cat reports/cron.log
```

Expected: cron.log에 "실행 시작" → "python 종료 코드: 0" → (커밋 발생) → "실행 종료". `git log --oneline -1`이 "Update daily report", `git status`에서 origin과 동기화 확인. `reports/latest.html` 존재.

- [ ] **Step 4: Test idempotent guard (재실행 시 스킵)**

```bash
./scripts/run_daily_report.sh
tail -5 reports/cron.log
```

Expected: "오늘(...) 이미 생성됨 — 스킵" 로그, 새 커밋 없음

- [ ] **Step 5: Commit**

```bash
git add scripts/run_daily_report.sh .gitignore
git commit -m "Add daily report cron wrapper"
```

---

### Task 5: crontab 등록 및 cron 서비스 확인

**Files:**
- Modify: crontab (사용자 crontab, 파일 아님)

**Interfaces:**
- Consumes: Task 4의 `scripts/run_daily_report.sh`
- Produces: 매일 8:10/12:10/16:10 KST 자동 실행

- [ ] **Step 1: Add crontab entries (기존 ETF Scout 항목 보존)**

```bash
(crontab -l; cat <<'EOF'

# Trend Dashboard 일일 리포트 — 하루 3번 시도(8:10/12:10/16:10 KST).
# 멱등 가드(run_daily_report.sh: 오늘치 reports/report_<날짜>.html 있으면 스킵).
# ETF Scout(정각)과 10분 어긋나게 배치해 동시 실행을 피한다.
10 8 * * * /home/byoun/projects/trend-dashboard/scripts/run_daily_report.sh
10 12 * * * /home/byoun/projects/trend-dashboard/scripts/run_daily_report.sh
10 16 * * * /home/byoun/projects/trend-dashboard/scripts/run_daily_report.sh
EOF
) | crontab -
```

- [ ] **Step 2: Verify crontab**

Run: `crontab -l`
Expected: 기존 ETF Scout 3줄 + 신규 3줄 (`10 8`, `10 12`, `10 16`) 모두 존재

- [ ] **Step 3: Verify cron daemon is running**

Run: `service cron status || pgrep -a cron`
Expected: cron 실행 중 (ETF Scout이 이미 cron으로 운영 중이므로 실행 중이어야 정상. 아니면 `sudo service cron start` 후 사용자에게 보고)

- [ ] **Step 4: Final verification**

Run: `/home/byoun/miniconda3/bin/python3 -m pytest tests/ -v && git status --short`
Expected: 2 passed, 워킹트리 깨끗함

---

## Self-Review Notes

- 스펙 10개 섹션 → Task 2 템플릿에 전부 매핑됨 (기간정보/KPI/NAV/DD/Current Signal/Raw Signal/연간/월간/보유종목/Stats — Stats는 plotly 테이블 + 포맷된 df 테이블 둘 다)
- `run_top_k_backtest`가 반환하는 BM NAV는 tcost=0 (함수 내부 고정) — 대시보드와 동일
- `wgt.index[-1] <= navs.index[-1]` 테스트: `.loc[:display_end]` 절단으로 보장
- Task 3 커밋에 오늘 리포트가 포함되고 Task 4 Step 3에서 삭제 후 재생성 — 커밋/푸시 경로 검증 목적
