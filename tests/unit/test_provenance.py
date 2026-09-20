import pytest
from pathlib import Path
from datapilot.ledger.store import LedgerStore
from datapilot.ledger.provenance import ProvenanceDAG

def test_provenance_dag(tmp_path: Path):
    store = LedgerStore(tmp_path / "ledger.db")
    # ev_0001 (plan)
    store.record_evidence("run_1", "plan", "planner", "h", "c", {}, {}, ["col_a"], "ok")
    # ev_0002 (sql, depends on ev_0001)
    store.record_evidence("run_1", "sql", "sql", "h", "c", {}, {}, ["col_b"], "ok", depends_on=["ev_0001"])
    # ev_0003 (stats, depends on ev_0002)
    store.record_evidence("run_1", "stat_test", "stats", "h", "c", {}, {}, ["col_c"], "ok", depends_on=["ev_0002"])
    
    dag = ProvenanceDAG(store, "run_1")
    
    assert set(dag.ancestors("ev_0003")) == {"ev_0001", "ev_0002"}
    assert set(dag.descendants("ev_0001")) == {"ev_0002", "ev_0003"}
    
    columns = dag.trace_to_columns("ev_0003")
    assert columns == {"col_a", "col_b", "col_c"}
