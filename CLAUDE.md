# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trend Dashboard is a Python Dash application for analyzing momentum-based rotation strategies across ETFs. It downloads price data from Yahoo Finance, generates momentum signals, runs backtests, and displays interactive visualizations.

## Running the Application

```bash
gunicorn app:server
```
Runs on port 8000 by default.

## Architecture

**Data Flow:**
```
Price Data (Yahoo Finance) → Signal Generation → Portfolio Weights → Backtest → Visualization
```

**Module Organization (`src/`):**
- `analysis.py` - `run_analysis()`: everything the Run Analysis button computes (pure pandas, raises `AnalysisError`, returns warnings as strings). Used by the dashboard callback and the report. Also `build_composite_bm()` for Custom BM
- `config.py` - Ticker presets (5; default `Alpha (Default)` = 20 ETFs), window parameters, weight methods, sidebar defaults (`DEFAULT_*`, read by `layout.py` and the report), `IMPLEMENTATION_LAG` (business days from month-end signal to rebalance; default 1), `PERIODS_PER_YEAR` (261: prices are on every weekday)
- `data.py` - `load_price_data()` downloads and aligns daily prices from yfinance
- `signals.py` - `generate_signal()` computes multi-window momentum rankings (cross-sectional)
- `portfolio.py` - `compute_weight_top_k()`, `compute_weight_quantile()`, and `backtest()` engine
- `metrics.py` - `summary_stats()` for CAGR, Sharpe, MDD; `calculate_kpis()` for dashboard display; `format_stats()` / `kpi_lines()` display strings shared by dashboard and report; KPI YTD/MTD start from the previous year-/month-end close (same base as `annual_returns()` / `monthly_returns()`)
- `charts.py` - Plotly visualizations: NAV, drawdown, heatmaps, signal tables (`signal_quantile_columns()` is shared with the report)
- `optimizer.py` - `optimize_sharpe()` grid search for optimal lookback windows and weights

**Dash Components (`dash_components/`):** split by screen area; `*_layout.py` = what is shown, `*_callbacks.py` = behaviour
- `layout.py` - `create_layout()`: stores, downloads, sidebar + results area
- `sidebar_layout.py` - sidebar, one function per section (`_signal_section()`, `_benchmark_section()`, ...); also help labels and the core/BM row builders (reused by `sidebar_callbacks.py`)
- `results_layout.py` - results area, one function per section (`_holdings_section()`, ...) + user-guide texts
- `callbacks.py` - `register_callbacks()` wires the three modules below
- `sidebar_callbacks.py` - collapse toggles, ticker selection, core/BM rows
- `results_callbacks.py` - Run Analysis (wraps `src.analysis.run_analysis()`), charts, heatmaps, Excel downloads
- `optimization_callbacks.py` - Optimize and walk-forward

**Entry Point:**
- `app.py` - Dash app, uses gunicorn for production

**Manual Report (`scripts/`):**
- `generate_report.py` - Calls `src.analysis.run_analysis()` with the sidebar defaults and renders `reports/report_<date>.html` (run manually: `python scripts/generate_report.py`; `reports/` is gitignored, no cron/auto-commit)
- `report_charts.py` - Static matplotlib charts used by the report

## Key Patterns

**Adding a new signal type:** Create function in `signals.py` returning a DataFrame of rankings, use it in `analysis.run_analysis()`, and give the optimizer an equivalent fast path (it builds signals from `precompute_momentums()` / `combine_signals()`; `tests/test_optimizer_consistency.py` pins the two together)

**Adding a new chart:** Create function in `charts.py` using Plotly, return a `go.Figure`, call from callbacks

**Adding new metrics:** Add calculation to `metrics.py`, reference in callbacks

**Adding a strategy variant:** Implement weight computation in `portfolio.py` following `compute_weight_top_k()` pattern

**Rebalance timing:** signal weights are indexed by month-end signal date; move them to rebalance dates only via `portfolio.to_rebalance_dates()` / `rebalance_dates()` (reads `config.IMPLEMENTATION_LAG`). Never write `+ BMonthBegin(1)` / `+ BDay(n)` elsewhere — a test scans for it. Holidays are intentionally not skipped. Holdings charts label columns by `portfolio.holding_months()` (the month after the signal, for any lag), not by the rebalance date.

**Benchmark & core-satellite:** BM = `portfolio.equal_weight_benchmark(signal)` (equal weight across tickers that have a signal — the strategy's own candidate set). Core blending = `portfolio.blend_core_satellite()`. Core-Satellite applies to Top-K only: in Quantile mode `run_analysis()` drops the core (with a notice) so buckets and BM stay core-free. `optimize_sharpe()` / `walk_forward_optimize()` take `universe=` (selectable tickers): in core-satellite mode `price` also carries core-only tickers, which must never enter the signal.

**Annualisation:** prices are resampled to every weekday (holidays forward-filled), so annualise only with `config.PERIODS_PER_YEAR` (261) — `metrics.summary_stats()` / `metrics.sharpe_from_nav()`. Never hard-code 252.

**Walk-forward:** fold edges are snapped to rebalance dates (`_holding_month_start()`), and the out-of-sample curve is ONE continuous backtest of the folds' weights concatenated (a fold switch is a normal rebalance with turnover cost) — never chain per-fold NAVs.

**Backtest rules live only in `portfolio.py`** (`compute_weight_top_k()`, `backtest()`). `optimizer.py` calls them via `_strategy_nav()` / `_benchmark_nav()` — never re-implement the weight or NAV loop there.

**Sidebar inputs:** never swap a blank or invalid value for a default. Validate with `analysis.check_whole_number()` / `analysis.parse_core_config()` (raise `AnalysisError` or show a warning), and give each button's `disabled` exactly ONE writer callback that checks all of its conditions (ticker count + inputs); otherwise callbacks overwrite each other.

**Adding Dash callbacks:** Add inside `register(app)` of the matching `dash_components/*_callbacks.py`; new UI goes into the matching section function in `sidebar_layout.py` / `results_layout.py`. Keep `.py` file names unique across the project (no second `analysis.py` etc.)

## Code Style

- Python 3.10+ with type hints using union syntax (`pd.DataFrame | pd.Series`)
- Pandas for data manipulation, Plotly for charts
- Do not add global `warnings.filterwarnings('ignore')` — warnings must stay visible (a global filter hid pandas deprecations before)

## Testing

```bash
python -m pytest
```
- Network-free (synthetic prices), ~20s. pytest is a dev tool and is not in `requirements.txt`.
- `pytest.ini` turns `FutureWarning` into errors so pandas deprecations fail fast.
- `tests/test_optimizer_consistency.py` checks the optimizer scores parameters exactly like the dashboard: momentum cache (`combine_signals`) == `generate_signal`, and train/test windows == `portfolio.backtest` run on that window with full-history volatility.
- `tests/test_golden.py` snapshots end-to-end numbers in `tests/golden/core_outputs.json`. After an *intentional* math change: `UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py`, then review the JSON diff before committing.
