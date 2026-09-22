"""
Tests for Phase 7: Multi-turn Session Manager.
"""
import json
import pytest
import pandas as pd
from datapilot.llm.client import MockLLMClient
from datapilot.ledger.store import LedgerStore
from datapilot.graph.workflow import DataPilotWorkflow
from datapilot.session.followup import SessionManager


@pytest.fixture
def session_env(tmp_path):
    store = LedgerStore(db_path=tmp_path / "ledger.db")
    snap_base = tmp_path / "snapshots"
    dataset_hash = "sesshash123"
    snap_dir = snap_base / dataset_hash
    snap_dir.mkdir(parents=True)

    df = pd.DataFrame({
        "revenue_before": [100.0, 110.0, 120.0] * 12,
        "revenue_after": [130.0, 140.0, 150.0] * 12,
        "region": ["NA", "EU", "APAC"] * 12,
    })
    df.to_parquet(snap_dir / "sales.parquet")

    mock_llm = MockLLMClient()

    mock_plan = json.dumps({
        "hypotheses": [
            {
                "id": "H1",
                "statement": "Revenue increased significantly",
                "columns": ["revenue_before", "revenue_after"],
                "test_type": "before_after",
            }
        ],
        "plan": [
            {
                "step": 1,
                "agent": "stats",
                "action": "Run before_after",
                "hypothesis_id": "H1",
            }
        ],
    })
    mock_llm.register("Planner", mock_plan)

    mock_stats = json.dumps({
        "test": "before_after",
        "columns": {"a": "revenue_before", "b": "revenue_after"},
        "justification": "Compare revenue before and after",
    })
    mock_llm.register("Stats Agent", mock_stats)

    mock_analyst = json.dumps({
        "conclusions": [
            {
                "hypothesis_id": "H1",
                "verdict": "supported",
                "confidence": "high",
                "summary": "Revenue increased significantly.",
                "caveats": [],
                "evidence_ids": ["ev_0002"],
            }
        ],
        "overall_summary": "Overall growth confirmed.",
    })
    mock_llm.register("Lead Analyst", mock_analyst)

    mock_viz = json.dumps({
        "charts": []
    })
    mock_llm.register("Visualization Agent", mock_viz)

    wf = DataPilotWorkflow(llm=mock_llm, store=store, snapshot_dir=snap_base)
    schema = {"sales": {"revenue_before": "DOUBLE", "revenue_after": "DOUBLE", "region": "VARCHAR"}}
    role_map = {"revenue_before": "metric", "revenue_after": "metric", "region": "category"}

    session = SessionManager(
        workflow=wf,
        dataset_hash=dataset_hash,
        schema=schema,
        role_map=role_map,
        table_name="sales",
    )
    return session, store


def test_session_multi_turn(session_env):
    session, store = session_env

    # Turn 1
    t1 = session.ask("Did revenue increase after the price change?")
    assert t1["turn_index"] == 1
    assert t1["run_id"].endswith("_t1")
    assert len(t1["evidence_ids"]) >= 5
    assert len(session.turns) == 1

    # Turn 2: Follow-up question
    t2 = session.ask("What is the effect in the EU region specifically?")
    assert t2["turn_index"] == 2
    assert t2["run_id"].endswith("_t2")
    assert len(session.turns) == 2

    # Verify session history and cumulative evidence DAG
    history = session.get_session_history()
    assert len(history) == 2
    assert history[0]["turn_index"] == 1
    assert history[1]["turn_index"] == 2

    # Cumulative evidence list should grow across turns
    assert len(session.all_evidence_ids) >= len(t1["evidence_ids"])

    # Ledger integrity should remain 100% valid
    assert store.verify_chain(t1["run_id"]) is True
    assert store.verify_chain(t2["run_id"]) is True
