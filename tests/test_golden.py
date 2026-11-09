"""Golden regression: end-to-end numbers on a fixed synthetic universe.

Any change to signal / weight / backtest / metrics / optimizer math shows up here.
If a change is intentional, regenerate and review the diff of the JSON:

    UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py
"""
import json
import math
import os
from pathlib import Path

import pandas as pd

from src.metrics import summary_stats
from src.optimizer import optimize_sharpe, walk_forward_optimize
from src.portfolio import run_quantile_backtest, run_top_k_backtest
from src.signals import generate_signal

from conftest import make_prices

GOLDEN = Path(__file__).parent / "golden" / "core_outputs.json"
PARAMS = dict(short_window=1, mid_window=3, long_window=11, short_wgt=0.1, mid_wgt=0.1, long_wgt=0.8)


def _monthly(nav: pd.Series) -> dict[str, float]:
    m = nav.resample("BME").last()
    return {d.strftime("%Y-%m"): float(v) for d, v in m.items()}


def _scalar(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def compute_outputs() -> dict:
    price = make_prices(late_tickers=3)
    sig = generate_signal(price, **PARAMS, thresh=9)
    out = {}

    for wm in ("equal", "inverse_vol", "rank"):
        nav, bm, turnover = run_top_k_backtest(price, sig, top_k=5, weight_method=wm, tcost=0.002)
        out[f"top5_{wm}_nav"] = _monthly(nav)
        out[f"top5_{wm}_turnover_sum"] = float(turnover.sum())
        if wm == "equal":
            out["bm_nav"] = _monthly(bm)
            stats = summary_stats(pd.DataFrame({"strat": nav, "bm": bm}))
            out["stats"] = {f"{c}.{r}": float(stats.loc[r, c]) for c in stats.columns for r in stats.index}

    q_nav, _, _ = run_quantile_backtest(price, sig, n_quantiles=5, tcost=0.002)
    out["quantile_final_nav"] = {q: float(q_nav[q].iloc[-1]) for q in q_nav.columns}

    opt = optimize_sharpe(price, top_k=5, n_samples=8, seed=1)
    out["optimize"] = {"best_params": opt["best_params"], "best_sharpe": _scalar(opt["best_sharpe"])}

    wf = walk_forward_optimize(price, train_months=24, test_months=12, step_months=12, n_samples=4, seed=42)
    out["walk_forward"] = {
        "fold_params": [f["params"] for f in wf["folds"]],
        "oos_sharpe": _scalar(wf["oos_sharpe"]),
        "is_sharpe_avg": _scalar(wf["is_sharpe_avg"]),
        "final_params": wf["final_params"],
        "combined_oos_nav_last": _scalar(wf["combined_oos_nav"].iloc[-1]),
        "combined_oos_bm_nav_last": _scalar(wf["combined_oos_bm_nav"].iloc[-1]),
    }

    # Core-Satellite mode (weights blended with a fixed core) and single-ticker BM path
    core = {"tickers": ["T00", "T01"], "weights": [2, 1], "ratio": 0.6}
    opt_c = optimize_sharpe(price, top_k=5, weight_method="inverse_vol", tcost=0.002,
                            n_samples=8, seed=3, core_config=core)
    out["optimize_core"] = {"best_params": opt_c["best_params"], "best_sharpe": _scalar(opt_c["best_sharpe"])}
    wf_c = walk_forward_optimize(price, train_months=24, test_months=12, step_months=12, n_samples=4,
                                 seed=5, tcost=0.002, core_config=core, bm_ticker="T05", bm_data=price["T05"])
    out["walk_forward_core"] = {
        "oos_sharpe": _scalar(wf_c["oos_sharpe"]),
        "combined_oos_nav": _monthly(wf_c["combined_oos_nav"]),
        "combined_oos_bm_nav": _monthly(wf_c["combined_oos_bm_nav"]),
    }
    return json.loads(json.dumps(out, default=float))  # normalise numpy scalars


def _assert_close(actual, expected, path="root"):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), f"{path}: keys differ"
        for k in expected:
            _assert_close(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f"{path}: length differs"
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), f"{path}: {actual} != {expected}"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def test_core_outputs_match_golden():
    actual = compute_outputs()
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
    expected = json.loads(GOLDEN.read_text())
    _assert_close(actual, expected)
