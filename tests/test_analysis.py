"""src/analysis.run_analysis — the Run Analysis button's computation (also used by the report)."""
import numpy as np
import pandas as pd
import pytest

import src.analysis as analysis
from src.analysis import (
    AnalysisError,
    check_backtest_dates,
    check_tcost,
    check_whole_number,
    parse_core_config,
    parse_seed,
    run_analysis,
)
from src.metrics import format_stats, summary_stats
from src.optimizer import optimize_sharpe
from src.portfolio import (
    backtest,
    blend_core_satellite,
    compute_weight_top_k,
    run_top_k_backtest,
    to_rebalance_dates,
)
from src.signals import generate_signal, signal_history

from conftest import make_prices

UNIVERSE = [f"T{i:02d}" for i in range(12)]


@pytest.fixture
def market(monkeypatch):
    """Synthetic market served through the loader the analysis uses (12-ticker universe + SPY/AGG)."""
    prices = make_prices(late_tickers=3)
    extra = make_prices(n_tickers=2, seed=7).rename(columns={"T00": "SPY", "T01": "AGG"})
    allp = prices.join(extra)

    def load_price_data(tickers, start_date="2000-01-01", end_date=None, proxy=None):
        tickers = [tickers] if isinstance(tickers, str) else list(tickers)
        raw = allp.reindex(columns=tickers)[start_date:end_date]
        avail = [t for t in tickers if raw[t].notna().any()]
        return raw[avail].resample("B").last().ffill(), [t for t in tickers if t not in avail]

    monkeypatch.setattr(analysis, "load_price_data", load_price_data)
    return allp


def test_single_mode_defaults(market):
    res = run_analysis(UNIVERSE, backtest_start_date="2016-06-01", thresh=9)
    price = market[UNIVERSE]
    sig = generate_signal(price, 1, 3, 11, 0.1, 0.1, 0.8, thresh=9)
    nav, bm, _ = run_top_k_backtest(price, sig, top_k=5)

    assert list(res.navs.columns) == ["Top-5", "BM (Equal Weight)"]
    assert res.navs.index[0] == pd.Timestamp("2016-06-01")
    pd.testing.assert_series_equal(res.navs["Top-5"], nav["2016-06-01":], check_names=False, check_freq=False)
    pd.testing.assert_series_equal(res.navs["BM (Equal Weight)"], bm["2016-06-01":], check_names=False,
                                   check_freq=False)

    expected_w = to_rebalance_dates(compute_weight_top_k(price, sig, top_k=5)[0])
    pd.testing.assert_frame_equal(res.weights["Top-5"], expected_w.loc[: res.navs.index[-1]], check_freq=False)
    pd.testing.assert_frame_equal(res.signal, sig)  # full signal incl. the tentative last month
    assert res.warnings == [] and res.params["bm_col_name"] == "BM (Equal Weight)"


def test_display_window_and_end_date(market):
    res = run_analysis(UNIVERSE, backtest_start_date="2017-01-01", backtest_end_date="2018-06-30", thresh=9)
    assert res.navs.index[0] >= pd.Timestamp("2017-01-01")
    assert res.navs.index[-1] == pd.Timestamp("2018-06-29")
    assert res.weights["Top-5"].index[-1] <= pd.Timestamp("2018-06-29")
    assert res.params["display_end"] == "2018-06-29"


def test_quantile_mode(market):
    res = run_analysis(UNIVERSE, strategy_type="Quantile", n_quantiles=4, thresh=9)
    assert list(res.navs.columns) == ["Q1", "Q2", "Q3", "Q4", "BM (Equal Weight)"]
    assert set(res.weights) == {"Q1", "Q2", "Q3", "Q4"}


def test_more_quantiles_than_tickers_with_a_signal_is_an_error(market):
    """8 tickers cannot fill 10 quantiles: some would hold nothing, every month."""
    with pytest.raises(AnalysisError, match="Number of Quantiles"):
        run_analysis(UNIVERSE[:8], strategy_type="Quantile", n_quantiles=10, thresh=5)
    assert run_analysis(UNIVERSE[:8], strategy_type="Quantile", n_quantiles=8, thresh=5).navs.notna().all().all()


def test_empty_quantiles_before_the_backtest_start_do_not_spoil_the_results(market):
    """Before the 3 late tickers get a signal only 9 tickers rank, so Q10 is empty (cash) in
    those months. From 2018 all 12 rank: the shown results must be complete, not NaN."""
    res = run_analysis(UNIVERSE, strategy_type="Quantile", n_quantiles=10, thresh=9,
                       backtest_start_date="2018-01-01")
    assert res.navs.notna().all().all()
    assert len(summary_stats(res.navs).columns) == 11 and summary_stats(res.navs).loc["nsamples"].min() > 400


def test_quantile_mode_ignores_core_satellite(market):
    """Core-Satellite is Top-K only: quantile buckets and their benchmark stay core-free."""
    core = [{"ticker": "SPY", "weight": 60}, {"ticker": "AGG", "weight": 40}]
    res = run_analysis(UNIVERSE, strategy_type="Quantile", strategy_mode="core-satellite", core_config=core, thresh=9)
    plain = run_analysis(UNIVERSE, strategy_type="Quantile", thresh=9)

    assert list(res.navs.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5", "BM (Equal Weight)"]
    pd.testing.assert_frame_equal(res.navs, plain.navs)
    assert res.warnings == ["Core-Satellite applies to Top-K only. Quantile results exclude the core."]
    assert res.params["core_ratio"] is None and res.params["strategy_mode"] == "single"
    # no core config needed in quantile mode
    assert run_analysis(UNIVERSE, strategy_type="Quantile", strategy_mode="core-satellite", thresh=9).navs.shape


def test_core_satellite_mode(market):
    core = [{"ticker": "SPY", "weight": 60}, {"ticker": "AGG", "weight": 40}]
    res = run_analysis(UNIVERSE, strategy_mode="core-satellite", core_weight_pct=70, core_config=core,
                       tcost_pct=0.1, thresh=9)

    combined, core_col, sat_col, bm_col = res.navs.columns
    assert combined == "Combined (Core 70% / Satellite 30%)"
    assert bm_col == "BM (Core 70% + EW 30%)"

    sig = generate_signal(market[UNIVERSE], 1, 3, 11, 0.1, 0.1, 0.8, thresh=9)
    sat, _ = compute_weight_top_k(market[UNIVERSE], sig, top_k=5)
    blended = blend_core_satellite(sat, market, ["SPY", "AGG"], [60, 40], 0.7)
    nav, _ = backtest(market, to_rebalance_dates(blended), tcost=0.001)
    pd.testing.assert_series_equal(res.navs[combined], nav.loc[res.navs.index], check_names=False,
                                   check_freq=False)
    assert not {"SPY", "AGG"} & set(res.weights["Top-5"].columns)  # core tickers never in the satellite


def test_custom_benchmark(market):
    res = run_analysis(UNIVERSE, bm_type="custom", bm_config=[{"ticker": "SPY", "weight": 100}], thresh=9)
    bm = res.navs["BM (SPY 100.0%)"]
    spy = market["SPY"].loc[res.navs.index]
    np.testing.assert_allclose(bm.values, (spy / spy.iloc[0] * 1000).values, rtol=1e-12)


def test_errors_and_warnings(market):
    with pytest.raises(AnalysisError, match="requires core tickers"):
        run_analysis(UNIVERSE, strategy_mode="core-satellite", core_config=[])
    with pytest.raises(AnalysisError, match="Not enough valid tickers"):
        run_analysis(["T00", "NOPE"])

    res = run_analysis(UNIVERSE + ["NOPE"], bm_type="custom", bm_config=[{"ticker": "ZZZ", "weight": 100}],
                       thresh=9)
    assert res.warnings == ["Data not found for: NOPE", "BM ticker(s) not found: ZZZ",
                            "Custom BM could not be built. Using Equal Weight BM instead."]
    assert res.params["bm_col_name"] == "BM (Equal Weight)"


@pytest.mark.parametrize("start, end", [
    ("2016-06-01", None),          # starts on a rebalance date
    ("2016-06-15", None),          # mid-month: positions taken on 2016-06-01 are already held
    ("2016-06-01", "2018-06-30"),  # Backtest End Date: data after it must not be used
])
def test_optimizer_best_sharpe_is_what_the_dashboard_shows(market, start, end):
    """Optimize in core-satellite mode, then Run Analysis with the result: same Sharpe."""
    core = [{"ticker": "SPY", "weight": 60}, {"ticker": "AGG", "weight": 40}]
    opt = optimize_sharpe(market, top_k=5, weight_method="inverse_vol", tcost=0.002, start_date=start,
                          end_date=end, thresh=9, n_samples=6, seed=0, universe=UNIVERSE,
                          core_config={"tickers": ["SPY", "AGG"], "weights": [60, 40], "ratio": 0.7})
    p = opt["best_params"]
    res = run_analysis(UNIVERSE, backtest_start_date=start, backtest_end_date=end, thresh=9,
                       weight_method="inverse_vol", tcost_pct=0.2,
                       short_window=p["short_window"], mid_window=p["mid_window"], long_window=p["long_window"],
                       short_wgt=p["short_wgt"], mid_wgt=p["mid_wgt"], long_wgt=p["long_wgt"],
                       strategy_mode="core-satellite", core_weight_pct=70, core_config=core)

    shown = summary_stats(res.navs).loc["sharpe", "Combined (Core 70% / Satellite 30%)"]
    assert shown == pytest.approx(opt["best_sharpe"], rel=1e-9)


# ---------------------------------------------------------------- input validation

@pytest.mark.parametrize("value, problem", [
    (None, "Test period: enter a whole number of at least 1."), ("", "Test period: enter a whole number of at least 1."),
    (0, "Test period must be at least 1."),
    (2.5, "Test period must be a whole number."), ("x", "Test period must be a number."), (12, None), (12.0, None),
])
def test_check_whole_number(value, problem):
    assert check_whole_number(value, 1, "Test period") == problem


def test_parse_core_config():
    cfg = parse_core_config([{"ticker": " spy ", "weight": 60}, {"ticker": "AGG", "weight": 0}, {"ticker": "", "weight": 5}], 70)
    assert cfg == {"tickers": ["SPY"], "weights": [60.0], "ratio": 0.7}
    for pct in (0, 100, None, ""):
        with pytest.raises(AnalysisError, match="between 1 and 99"):
            parse_core_config([{"ticker": "SPY", "weight": 100}], pct)
    with pytest.raises(AnalysisError, match="requires core tickers"):
        parse_core_config([{"ticker": "SPY", "weight": 0}], 70)


def test_core_allocation_zero_is_an_error_not_70(market):
    core = [{"ticker": "SPY", "weight": 100}]
    with pytest.raises(AnalysisError, match="between 1 and 99"):
        run_analysis(UNIVERSE, strategy_mode="core-satellite", core_weight_pct=0, core_config=core, thresh=9)
    # quantile mode drops the core, so its settings are not validated there
    assert run_analysis(UNIVERSE, strategy_type="Quantile", strategy_mode="core-satellite", core_weight_pct=0,
                        core_config=core, thresh=9).warnings


def test_fewer_tickers_than_min_valid_tickers_is_an_error(market):
    with pytest.raises(AnalysisError, match="fewer than Min Valid Tickers"):
        run_analysis(UNIVERSE[:5] + ["NOPE"], thresh=9)


@pytest.mark.parametrize("value, ok", [(None, False), ("", False), (-0.1, False), (1.5, False), ("x", False),
                                       (0, True), (0.0, True), (0.1, True), (1, True)])
def test_check_tcost(value, ok):
    problem = check_tcost(value)
    assert (problem is None) == ok
    if not ok:
        assert problem == "Transaction cost: enter a number from 0 to 1 (%)."


def test_blank_tcost_is_an_error_not_zero_cost(market):
    """The number input sends None when blank or out of range: never run it as 0 % cost."""
    with pytest.raises(AnalysisError, match="Transaction cost"):
        run_analysis(UNIVERSE, tcost_pct=None, thresh=9)
    assert run_analysis(UNIVERSE, tcost_pct=0, thresh=9).params["tcost"] == 0.0


@pytest.mark.parametrize("start, end, problem", [
    ("2018-01-01", "2017-01-01", "Backtest End Date must be after Backtest Start Date."),
    ("2018-01-01", "2018-01-01", "Backtest End Date must be after Backtest Start Date."),
    ("2018-01-01", None, None),
    ("2018-01-01", "2018-02-01", None),
])
def test_check_backtest_dates(start, end, problem):
    assert check_backtest_dates(start, end) == problem


def test_end_date_before_start_is_an_error(market):
    with pytest.raises(AnalysisError, match="Backtest End Date must be after Backtest Start Date"):
        run_analysis(UNIVERSE, backtest_start_date="2018-01-01", backtest_end_date="2017-01-01", thresh=9)


def test_end_date_before_the_first_result_is_an_error(market):
    """Prices start 2015-01; the first signal needs 11 months: nothing to show up to 2015-06-30."""
    with pytest.raises(AnalysisError, match="No results up to Backtest End Date"):
        run_analysis(UNIVERSE, backtest_start_date="2015-01-01", backtest_end_date="2015-06-30", thresh=9)


def test_all_zero_window_weights_is_an_error(market):
    """All ranks would tie and the strategy would hold the first tickers in column order."""
    with pytest.raises(AnalysisError, match="window weights"):
        run_analysis(UNIVERSE, short_wgt=0, mid_wgt=0, long_wgt=0, thresh=9)


def test_duplicate_core_tickers_add_up():
    cfg = parse_core_config([{"ticker": "SPY", "weight": 50}, {"ticker": "agg", "weight": 20},
                             {"ticker": " spy", "weight": 30}], 70)
    assert cfg == {"tickers": ["SPY", "AGG"], "weights": [80.0, 20.0], "ratio": 0.7}


@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("  ", None), ("7", 7), (7, 7), ("0", 0)])
def test_parse_seed(value, expected):
    assert parse_seed(value) == expected


@pytest.mark.parametrize("value", ["abc", "-1", "1.5"])
def test_parse_seed_rejects_non_whole_numbers(value):
    with pytest.raises(AnalysisError, match="Seed"):
        parse_seed(value)


def test_custom_bm_without_tickers_says_so(market):
    res = run_analysis(UNIVERSE, bm_type="custom", bm_config=[], thresh=9)
    assert res.params["bm_col_name"] == "BM (Equal Weight)"
    assert "Custom BM has no tickers applied. Using Equal Weight BM instead." in res.warnings


# ---------------------------------------------------------------- shared display helpers

def test_format_stats_strings(market):
    res = run_analysis(UNIVERSE, thresh=9)
    f = format_stats(summary_stats(res.navs))  # must not warn (FutureWarning = error in pytest.ini)
    assert f.loc["cagr"].str.endswith("%").all() and f.loc["nsamples"].str.isdigit().all()


def test_signal_history_labels_tentative_month():
    idx = pd.to_datetime(["2026-08-31", "2026-09-30", "2026-10-30", "2026-11-30"])
    sig = pd.DataFrame({"A": [1, 2, 1, 2], "B": [2, 1, 2, 1]}, index=idx, dtype=float)
    table = signal_history(sig, data_end="2026-11-09", months=3)
    assert list(table.columns) == ["2026-09-30", "2026-10-30", "2026-11-09"]
    assert list(table.index) == ["A", "B"]
