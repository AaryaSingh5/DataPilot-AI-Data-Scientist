import pytest
from datapilot.ledger.replay import ReplayEngine

def test_compare_dicts():
    engine = ReplayEngine(store=None) # type: ignore
    
    d1 = {"a": 1, "b": 2.0, "c": "hello", "d": float("inf"), "e": float("nan")}
    d2 = {"a": 1, "b": 2.0000000001, "c": "hello", "d": float("inf"), "e": float("nan")}
    
    diffs = engine._compare_dicts(d1, d2)
    assert not diffs # Empty dict means no mismatches
    
    d3 = {"a": 2}
    diffs2 = engine._compare_dicts(d1, d3)
    assert "a" in diffs2
    assert diffs2["a"]["reason"] == "mismatch"
    assert "b" in diffs2
    assert diffs2["b"]["reason"] == "missing_in_actual"
    
    d4 = {"a": 1, "b": 2.0, "c": "hello", "d": float("inf"), "e": float("nan"), "f": "extra"}
    diffs3 = engine._compare_dicts(d1, d4)
    assert "f" in diffs3
    assert diffs3["f"]["reason"] == "missing_in_expected"
