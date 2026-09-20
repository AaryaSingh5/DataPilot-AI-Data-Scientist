"""Phase 4 deterministic tool tests."""
import math
import duckdb
import pytest
from datapilot.tools.viz import bar_chart, line_chart, scatter_chart, heatmap, waterfall_chart, VizError
from datapilot.tools.stats.correction import benjamini_hochberg
from datapilot.tools.stats.comparisons import before_after
from datapilot.tools.stats.association import association
from datapilot.tools.stats.power import power_analysis

# ── Viz tests ────────────────────────────────────────────────────────────────

def test_bar_chart_basic():
    data = [{"cat": "A", "val": 10}, {"cat": "B", "val": 20}]
    spec = bar_chart(data, x="cat", y="val", title="Test")
    assert spec["type"] == "bar"
    assert len(spec["series"]) == 2
    assert spec["series"][0]["label"] == "A"


def test_viz_rejects_nan():
    data = [{"x": 1.0, "y": float("nan")}]
    with pytest.raises(VizError, match="non-finite"):
        line_chart(data, x="x", y="y")


def test_heatmap_from_matrix():
    matrix = {"A": {"A": 1.0, "B": 0.5}, "B": {"A": 0.5, "B": 1.0}}
    spec = heatmap(matrix)
    assert spec["type"] == "heatmap"
    assert len(spec["cells"]) == 4


def test_waterfall_chart():
    contribs = {"customer_effect": 120.0, "arpu_effect": -30.0, "interaction": 5.0}
    spec = waterfall_chart(contribs)
    assert spec["type"] == "waterfall"
    assert len(spec["bars"]) == 3
    # Running total check
    last_end = spec["bars"][-1]["end"]
    expected = sum(contribs.values())
    assert abs(last_end - expected) < 1e-9


# ── Stats tests ───────────────────────────────────────────────────────────────

def test_bh_correction_monotonic():
    p_vals = [0.001, 0.01, 0.04, 0.2, 0.5]
    q_vals = benjamini_hochberg(p_vals)
    assert len(q_vals) == len(p_vals)
    # q-values must all be >= corresponding p-values
    for p, q in zip(p_vals, q_vals):
        assert q >= p - 1e-12


def test_bh_all_significant():
    p_vals = [0.001, 0.002, 0.003]
    q_vals = benjamini_hochberg(p_vals)
    assert all(q <= 0.05 for q in q_vals)


def test_before_after_normal():
    rng = __import__("numpy").random.default_rng(42)
    a = rng.normal(100, 10, 50).tolist()
    b = rng.normal(115, 10, 50).tolist()
    res = before_after(a, b)
    assert "p" in res
    assert res["p"] < 0.05   # should detect difference


def test_association_correlated():
    import numpy as np
    x = list(range(50))
    y = [xi * 2 + 1 + np.random.default_rng(0).normal(0, 2) for xi in x]
    res = association(x, y)
    assert res["pearson_r"] > 0.9


def test_power_analysis():
    res = power_analysis(effect_size=0.5, nobs1=50)
    assert 0 < res["power"] <= 1.0
