"""Dash wiring (layout lifecycle, single writers, where results are shown) and result charts."""
import inspect
import json
import pathlib
import re
from collections import Counter

import numpy as np
import pandas as pd
import pytest

import app as dash_app
from dash import dcc

from src.charts import create_quantile_holding_heatmap, create_quantile_spread_chart
from src.optimizer import walk_forward_optimize
from src.portfolio import compute_weight_quantile, to_rebalance_dates
from src.signals import generate_signal

from conftest import make_prices


@pytest.fixture(scope="module")
def dependencies() -> list[dict]:
    client = dash_app.app.server.test_client()
    return json.loads(client.get("/_dash-dependencies").data)


def _outputs(cb: dict) -> list[str]:
    """'..a.prop...b.prop@hash..' -> ['a.prop', 'b.prop']"""
    return [o.split("@")[0] for o in cb["output"].strip(".").split("...")]


def _find(component, cid):
    if getattr(component, "id", None) == cid:
        return component
    children = getattr(component, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None and hasattr(child, "to_plotly_json"):
            hit = _find(child, cid)
            if hit is not None:
                return hit
    return None


def test_layout_is_built_on_every_page_load():
    """Date pickers default to today: a layout built once at server start would freeze that day."""
    assert callable(dash_app.app.layout)


def test_every_button_disabled_has_exactly_one_writer(dependencies):
    """Two writers (e.g. a validator and a running= flag) overwrite each other."""
    writers = Counter()
    for cb in dependencies:
        props = _outputs(cb) + list((cb.get("running") or {}).get("running", {}))
        writers.update(p for p in props if p.endswith(".disabled"))
    assert writers, "no disabled writers found"
    assert {p: n for p, n in writers.items() if n != 1} == {}


def test_walk_forward_results_do_not_depend_on_run_analysis(dependencies):
    """Walk-forward can run before Run Analysis: its results must not sit inside the hidden
    analysis-results block, and Run Analysis must not hide them."""
    layout = dash_app.app.layout() if callable(dash_app.app.layout) else dash_app.app.layout
    assert _find(layout, "walk-forward-results-section") is not None
    assert _find(_find(layout, "analysis-results"), "walk-forward-results-section") is None

    run_cb = next(cb for cb in dependencies if "analysis-warnings.children" in _outputs(cb))
    assert "walk-forward-results-section.style" not in _outputs(run_cb)


def test_quantile_spread_starts_at_zero():
    """NAVs start at 1000 at the first signal, not at the display start: rebase before subtracting."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    q_nav = pd.DataFrame({"Q1": [1100.0, 1110, 1120, 1130, 1140], "Q5": [1500.0, 1530, 1560, 1590, 1620]}, index=idx)
    y = np.asarray(create_quantile_spread_chart(q_nav, top_q="Q5").data[0].y, dtype=float)
    assert y[0] == pytest.approx(0.0)
    assert y[-1] == pytest.approx(1620 / 1500 * 100 - 1140 / 1100 * 100)


def test_walk_forward_has_no_separate_step():
    """Step = test keeps out-of-sample windows back to back; other steps overlap or leave gaps."""
    assert "step_months" not in inspect.signature(walk_forward_optimize).parameters


@pytest.mark.parametrize("n_quantiles", range(2, 11))  # every value the slider allows
def test_quantile_holdings_heatmap_draws_every_quantile_count(n_quantiles):
    p = make_prices(n_tickers=12)
    sig = generate_signal(p)
    q_wgts, _ = compute_weight_quantile(p, sig, n_quantiles=n_quantiles)
    fig = create_quantile_holding_heatmap({q: to_rebalance_dates(w).iloc[-24:] for q, w in q_wgts.items()},
                                          signal=sig)
    colorscale = fig.data[0].colorscale
    assert len(colorscale) == 2 * n_quantiles
    assert len({c for _, c in colorscale}) == n_quantiles  # one distinct colour per quantile


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None and hasattr(child, "to_plotly_json"):
            yield from _walk(child)


def test_every_store_is_read_by_a_callback(dependencies):
    """A store nobody reads only ships data to the browser (it once held the full price history)."""
    layout = dash_app.app.layout()
    stores = {c.id for c in _walk(layout) if isinstance(c, dcc.Store)}
    read = {dep["id"] for cb in dependencies for dep in cb["inputs"] + cb["state"]}
    assert stores and stores - read == set()


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_functions_named_in_claude_md_exist():
    """CLAUDE.md guides future changes: every `function()` it names must exist in the code."""
    doc = (ROOT / "CLAUDE.md").read_text()
    names = {n.split(".")[-1] for n in re.findall(r"`([A-Za-z_][\w.]*)\(\)`", doc)}
    code = "\n".join(p.read_text() for d in ("src", "dash_components", "scripts") for p in (ROOT / d).glob("*.py"))
    code += (ROOT / "app.py").read_text()
    missing = {n for n in names if not re.search(rf"^\s*def {n}\(", code, re.M)}
    assert names and missing == set()


def test_background_cache_lives_in_the_project_folder():
    """`python /path/to/app.py` from another folder must not create ./cache there."""
    assert pathlib.Path(dash_app.cache.directory) == ROOT / "cache"
