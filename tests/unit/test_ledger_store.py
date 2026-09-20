import pytest
import sqlite3
import pytest
from pathlib import Path
from datapilot.ledger.store import LedgerStore
from datapilot.ledger.models import Evidence

def test_ledger_append_only(tmp_path: Path):
    store = LedgerStore(tmp_path / "ledger.db")
    
    # Record first evidence
    ev = store.record_evidence(
        run_id="run_1",
        kind="plan",
        produced_by="planner",
        dataset_hash="hash_a",
        code="foo()",
        params={"a": 1},
        result={"val": 42},
        columns=["col1"],
        status="ok"
    )
    
    assert ev.id == "ev_0001"
    assert ev.prev_hash == "0" * 64
    assert ev.hash != "0" * 64
    
    # Record second evidence
    ev2 = store.record_evidence(
        run_id="run_1",
        kind="sql",
        produced_by="sql",
        dataset_hash="hash_a",
        code="SELECT *",
        params={},
        result={"rows": 10},
        columns=["col1"],
        status="ok",
        depends_on=["ev_0001"]
    )
    
    assert ev2.id == "ev_0002"
    assert ev2.prev_hash == ev.hash
    
    # Verify chain
    assert store.verify_chain("run_1") is True

    # Try to tamper via raw SQLite (UPDATE)
    with pytest.raises(sqlite3.IntegrityError, match="Updates to evidence are not allowed"):
        store.conn.execute("UPDATE evidence SET kind = 'hacked' WHERE id = 'ev_0001'")

    # Try to tamper via raw SQLite (DELETE)
    with pytest.raises(sqlite3.IntegrityError, match="Deletions from evidence are not allowed"):
        store.conn.execute("DELETE FROM evidence WHERE id = 'ev_0001'")
        
def test_tamper_detection(tmp_path: Path):
    store = LedgerStore(tmp_path / "ledger.db")
    store.record_evidence("run_1", "plan", "system", "dhash", "code", {}, {"v": 1}, [], "ok")
    store.record_evidence("run_1", "sql", "system", "dhash", "code", {}, {"v": 2}, [], "ok")
    
    assert store.verify_chain("run_1") is True
    
    # Drop triggers temporarily to simulate external DB modification
    store.conn.execute("DROP TRIGGER prevent_evidence_update")
    store.conn.execute("UPDATE evidence SET result = '{\"v\": 999}' WHERE id = 'ev_0001'")
    
    # Chain verification should fail because the hash of ev_0001 doesn't match its content anymore
    assert store.verify_chain("run_1") is False

def test_hash_determinism(tmp_path: Path):
    store = LedgerStore(tmp_path / "ledger.db")
    ev1 = {
        "id": "ev_0001", "run_id": "r1", "kind": "k", "produced_by": "p",
        "depends_on": [], "dataset_hash": "d", "code": "c",
        "params": {"b": 2, "a": 1}, # Out of order keys
        "result": {}, "columns": [], "artifact_path": None, "artifact_hash": None,
        "lib_versions": {}, "status": "ok", "error": None, "created_at": "date",
        "prev_hash": "0"*64
    }
    
    ev2 = dict(ev1)
    ev2["params"] = {"a": 1, "b": 2} # Different order
    
    hash1 = store.compute_hash(ev1)
    hash2 = store.compute_hash(ev2)
    assert hash1 == hash2 # Must be sort-independent canonical hashing
