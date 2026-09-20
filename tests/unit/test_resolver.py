import pytest
from pathlib import Path
from datapilot.ledger.store import LedgerStore
from datapilot.ledger.resolver import SafeResolver, UnresolvedPlaceholder

def test_safe_resolver(tmp_path: Path):
    store = LedgerStore(tmp_path / "ledger.db")
    store.record_evidence(
        run_id="run_1", kind="stat_test", produced_by="stats", dataset_hash="h", code="",
        params={"alpha": 0.05}, result={"pct_change": -12.3456, "n": 100, "string_val": "hello"},
        columns=[], status="ok"
    )
    
    resolver = SafeResolver(store)
    
    # Valid numeric format
    assert resolver.resolve_text("Change was {ev_0001.result.pct_change:.1f}%") == "Change was -12.3%"
    
    # Missing format specifier (default string conversion)
    assert resolver.resolve_text("N = {ev_0001.result.n}") == "N = 100"
    
    # Valid string specifier
    assert resolver.resolve_text("String is {ev_0001.result.string_val:s}") == "String is hello"
    
    # Invalid path
    with pytest.raises(UnresolvedPlaceholder, match="Path 'missing' not found"):
        resolver.resolve_text("Change was {ev_0001.result.missing}")
        
    # Invalid base path
    with pytest.raises(UnresolvedPlaceholder, match="must start with result. or params."):
        resolver.resolve_text("Change was {ev_0001.status}")
        
    # Non-numeric formatting
    with pytest.raises(UnresolvedPlaceholder, match="is not numeric"):
        resolver.resolve_text("Change was {ev_0001.result.string_val:.2f}")
        
    # Malicious format specifier
    with pytest.raises(UnresolvedPlaceholder, match="Format spec '.__class__' is not allowed"):
        resolver.resolve_text("{ev_0001.result.pct_change:.__class__}")

def test_safe_resolver_error_evidence(tmp_path: Path):
    store = LedgerStore(tmp_path / "ledger.db")
    store.record_evidence(
        run_id="run_1", kind="sql", produced_by="sql", dataset_hash="h", code="",
        params={}, result={}, columns=[], status="error", error="Failed"
    )
    
    resolver = SafeResolver(store)
    with pytest.raises(UnresolvedPlaceholder, match="has error status"):
        resolver.resolve_text("Value is {ev_0001.result.val}")
