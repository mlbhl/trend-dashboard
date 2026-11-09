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
- `config.py` - Ticker presets (5; default `Alpha (Default)` = 20 ETFs), window parameters, weight methods, `IMPLEMENTATION_LAG` (business days from month-end signal to rebalance; default 1)
- `data.py` - `load_price_data()` downloads and aligns daily prices from yfinance
- `signals.py` - `generate_signal()` computes multi-window momentum rankings (cross-sectional)
- `portfolio.py` - `compute_weight_top_k()`, `compute_weight_quantile()`, and `backtest()` engine
- `metrics.py` - `summary_stats()` for CAGR, Sharpe, MDD; `calculate_kpis()` for dashboard display
- `charts.py` - Plotly visualizations: NAV, drawdown, allocation, heatmap, signal tables
- `optimizer.py` - `optimize_sharpe()` grid search for optimal lookback windows and weights

**Dash Components (`dash_components/`):**
- `layout.py` - UI layout with sidebar controls and main content tabs
- `callbacks.py` - All Dash callbacks for interactivity and data processing; includes `_build_composite_bm()` for Custom BM

**Entry Point:**
- `app.py` - Dash app, uses gunicorn for production

**Manual Report (`scripts/`):**
- `generate_report.py` - Headless run of the dashboard's default analysis → `reports/report_<date>.html` (run manually: `python scripts/generate_report.py`; `reports/` is gitignored, no cron/auto-commit)
- `report_charts.py` - Static matplotlib charts used by the report

## Key Patterns

**Adding a new signal type:** Create function in `signals.py` returning a DataFrame of rankings, then integrate in callbacks

**Adding a new chart:** Create function in `charts.py` using Plotly, return a `go.Figure`, call from callbacks

**Adding new metrics:** Add calculation to `metrics.py`, reference in callbacks

**Adding a strategy variant:** Implement weight computation in `portfolio.py` following `compute_weight_top_k()` pattern

**Rebalance timing:** signal weights are indexed by month-end signal date; move them to rebalance dates only via `portfolio.to_rebalance_dates()` / `rebalance_dates()` (reads `config.IMPLEMENTATION_LAG`). Never write `+ BMonthBegin(1)` / `+ BDay(n)` elsewhere — a test scans for it. Holidays are intentionally not skipped.

**Benchmark & core-satellite:** BM = `portfolio.equal_weight_benchmark(signal)` (equal weight across tickers that have a signal — the strategy's own candidate set). Core blending = `portfolio.blend_core_satellite()`. `optimize_sharpe()` / `walk_forward_optimize()` take `universe=` (selectable tickers): in core-satellite mode `price` also carries core-only tickers, which must never enter the signal.

**Backtest rules live only in `portfolio.py`** (`compute_weight_top_k()`, `backtest()`). `optimizer.py` calls them via `_strategy_nav()` / `_benchmark_nav()` — never re-implement the weight or NAV loop there.

**Adding Dash callbacks:** Add to `dash_components/callbacks.py` within `register_callbacks()` function

## Code Style

- Python 3.10+ with type hints using union syntax (`pd.DataFrame | pd.Series`)
- Pandas for data manipulation, Plotly for charts
- Do not add global `warnings.filterwarnings('ignore')` — warnings must stay visible (a global filter hid pandas deprecations before)

## Testing

```bash
python -m pytest
```
- Network-free (synthetic prices), ~5s. pytest is a dev tool and is not in `requirements.txt`.
- `pytest.ini` turns `FutureWarning` into errors so pandas deprecations fail fast.
- `tests/test_optimizer_consistency.py` checks the optimizer scores parameters exactly like the dashboard: momentum cache (`combine_signals`) == `generate_signal`, and train/test windows == `portfolio.backtest` run on that window with full-history volatility.
- `tests/test_golden.py` snapshots end-to-end numbers in `tests/golden/core_outputs.json`. After an *intentional* math change: `UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py`, then review the JSON diff before committing.
