# Daily HTML Report — Design

**Date:** 2026-07-03
**Status:** Approved

## Goal

매일 오전(KST) 로컬 WSL에서 기본 설정으로 분석을 실행하고, 웹페이지 오른편에
표시되는 결과 전체를 self-contained HTML 리포트로 저장한다. Dash 서버나
브라우저 없이 `src/` 함수를 직접 호출한다.

## Components

### 1. Report generator — `scripts/generate_report.py` (new)

Dash 콜백 `run_analysis`의 디폴트 경로를 headless로 재현한다.

**Parameters (sidebar defaults와 동일):**

| Parameter | Value |
|---|---|
| Universe | Alpha (Default) preset, 20 tickers |
| Strategy mode | Single |
| Strategy type | Top-K, K=5, select_all=False |
| Windows | short=1, mid=3, long=11 (months) |
| Window weights | 0.10 / 0.10 / 0.80 (custom mode) |
| Signal thresh | 10 |
| Weight method | equal |
| Transaction cost | 0.00% |
| Benchmark | Equal Weight |
| Data start | 2000-01-01 |
| Backtest start | 2015-01-01 |
| Backtest end | today |

**Flow:**

```
load_price_data → generate_signal → run_top_k_backtest
→ compute_weight_top_k (holdings heatmap용)
→ calculate_kpis + src/charts.py 차트 함수들
→ HTML 렌더링
```

`run_analysis`와 동일하게 처리: window weight 정규화, BM equal-weight NAV 결합,
backtest start/end 기간 절단, weights의 BMonthBegin(1) shift 및 display_end 절단,
Current Signal용 tentative 시그널 행 유지.

**Report contents (웹페이지 오른편 결과 전체):**

1. 생성 시각 타임스탬프 + 백테스트 기간 정보
2. KPI 카드 (전략/BM별 CAGR, Vol, Sharpe, MDD, YTD) — HTML 테이블
3. NAV 차트 (rebased to 100)
4. Drawdown 차트
5. Current Signal 테이블 (signal date 표기 포함)
6. Raw Signal 최근 5일 테이블
7. 연간 수익률 차트
8. 월간 수익률 히트맵
9. 보유종목(holdings) 히트맵
10. Stats 테이블 (summary_stats)

**HTML output:** Plotly 차트는 `fig.to_html(full_html=False)`로 embed,
plotly.js는 CDN 로드(`include_plotlyjs="cdn"`, 첫 차트에만) — 리포트를 git에
커밋하므로 파일당 ~4MB(전체 embed) 대신 수백 KB로 유지한다.
테이블은 pandas `to_html` + 간단한 inline CSS.

### 2. Runner — `scripts/run_daily_report.sh` (new)

ETF Scout의 `run_daily.sh` 패턴을 따른다:

- cron 최소 환경 대응: `HOME`, `PATH` 명시
- **멱등 가드:** `reports/report_$(date +%F).html` 존재 시 스킵
- `/home/byoun/miniconda3/bin/python3` (base env, 의존성 확인됨)으로
  `scripts/generate_report.py` 실행
- 성공 시 `reports/latest.html`로 복사
- **git 커밋/푸시:** `git add reports` → 변경 있으면 커밋(단일 라인 메시지
  "Update daily report") 후 `git push` (ETF Scout `run_daily.sh` 패턴).
  푸시 실패(원격 diverge 등) 시 로그에 남기고 종료 — 리포트 파일 자체는 보존됨
- stdout/stderr를 `reports/cron.log`에 append (시작/종료 타임스탬프 포함)

### 3. Storage — `reports/`

- `reports/report_YYYY-MM-DD.html` — 날짜별 누적, **git 커밋/푸시 대상**
- `reports/latest.html` — 최신본 덮어쓰기, git 커밋/푸시 대상
- `reports/cron.log` — 실행 로그, `.gitignore`에 추가 (로그는 커밋하지 않음)

### 4. Scheduling — WSL cron

기존 ETF Scout(정각)과 10분 어긋나게, 동일한 3회 시도 패턴:

```cron
10 8 * * *  /home/byoun/projects/trend-dashboard/scripts/run_daily_report.sh
10 12 * * * /home/byoun/projects/trend-dashboard/scripts/run_daily_report.sh
10 16 * * * /home/byoun/projects/trend-dashboard/scripts/run_daily_report.sh
```

- 멱등 가드 덕분에 하루 1회만 실제 실행; 8:10에 PC가 꺼져 있으면 12:10/16:10에 따라잡음
- WSL은 Windows 로컬 시간(KST)을 따르므로 TZ 변환 불필요
- cron 서비스가 부팅 시 시작되는지 확인 (ETF Scout이 이미 cron으로 운영 중이므로
  이미 활성화되어 있을 가능성 높음; 아니면 설정)

## Error handling

- yfinance 다운로드 실패 등 예외 발생 시 traceback을 `cron.log`에 기록하고 비정상 종료
  → 당일 리포트 파일이 생성되지 않으므로 다음 시각(12:10/16:10)에 자동 재시도
- 일부 티커 데이터 누락 시 `run_analysis`와 동일하게 경고를 리포트 상단에 표기하고 계속 진행

## Testing

- 스크립트를 수동 실행해 HTML이 생성되고 브라우저에서 모든 섹션이 표시되는지 확인
- 멱등 가드 동작 확인 (같은 날 재실행 시 스킵)
- 대시보드를 직접 실행해 Run Analysis 결과와 리포트의 KPI 수치가 일치하는지 대조

## Out of scope

- Core-Satellite / Custom BM / Quantile 모드 리포트 (디폴트 설정만)
- 리포트 자동 열기, 이메일/알림 발송
- 과거 리포트 자동 정리(rotation)
