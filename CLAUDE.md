# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trend Dashboard is a Python Streamlit application for analyzing momentum-based rotation strategies across ETFs. It downloads price data from Yahoo Finance, generates momentum signals, runs backtests, and displays interactive visualizations.

## Running the Application

```bash
streamlit run app.py
```

The app runs on port 8501 by default.

## Architecture

**Data Flow:**
```
Price Data (Yahoo Finance) → Signal Generation → Portfolio Weights → Backtest → Visualization
```

**Module Organization (`src/`):**
- `config.py` - Default ticker universe (24 ETFs), window parameters, weight methods
- `data.py` - `load_price_data()` downloads and aligns daily prices from yfinance
- `signals.py` - `generate_signal()` computes multi-window momentum rankings (cross-sectional)
- `portfolio.py` - `compute_weight_top_k()`, `compute_weight_quantile()`, and `backtest()` engine
- `metrics.py` - `summary_stats()` for CAGR, Sharpe, MDD; `calculate_kpis()` for dashboard display
- `charts.py` - Plotly visualizations: NAV, drawdown, allocation, heatmap, signal tables
- `optimizer.py` - `optimize_sharpe()` grid search for optimal lookback windows and weights

**Main Entry Point:**
- `app.py` - Streamlit UI with sidebar controls, session state management, and analysis orchestration

## Key Patterns

**Adding a new signal type:** Create function in `signals.py` returning a DataFrame of rankings, then integrate in `app.py`

**Adding a new chart:** Create function in `charts.py` using Plotly, return a `go.Figure`, call from `app.py`

**Adding new metrics:** Add calculation to `metrics.py`, reference in `app.py` display sections

**Adding a strategy variant:** Implement weight computation in `portfolio.py` following `compute_weight_top_k()` pattern

## Code Style

- Python 3.10+ with type hints using union syntax (`pd.DataFrame | pd.Series`)
- Pandas for data manipulation, Plotly for charts
- No test suite currently configured
