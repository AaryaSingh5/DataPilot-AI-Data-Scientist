"""
Unit & integration tests for Phase 7: LangGraph Workflow.
"""
import json
import pytest
import pandas as pd
from pathlib import Path

from datapilot.llm.client import MockLLMClient
from datapilot.ledger.store import LedgerStore
from datapilot.graph.workflow import DataPilotWorkflow
from datapilot.graph.state import DataPilotState


@pytest.fixture
def workflow_env(tmp_path):
    store = LedgerStore(db_path=tmp_path / "ledger.db")
    snap_base = tmp_path / "snapshots"
    dataset_hash = "dathash123"
    snap_dir = snap_base / dataset_hash
    snap_dir.mkdir(parents=True)

    # Sample dataset
    df = pd.DataFrame({
        "revenue_before": [100.0, 110.0, 105.0, 115.0, 120.0] * 7,
        "revenue_after": [130.0, 140.0, 135.0, 145.0, 150.0] * 7,
        "region": ["NA", "EU", "NA", "EU", "NA"] * 7,
    })
    df.to_parquet(snap_dir / "sales.parquet")

    mock_llm = MockLLMClient()

    # Register mocks
    mock_plan = json.dumps({
        "hypotheses": [
            {
                "id": "H1",
                "statement": "Revenue increased significantly after change",
                "columns": ["revenue_before", "revenue_after"],
                "test_type": "before_after",
            }
        ],
        "plan": [
            {
                "step": 1,
                "agent": "stats",
                "action": "Run before_after comparison on revenue",
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
                "summary": "Revenue increased significantly across the sample.",
                "caveats": [],
                "evidence_ids": ["ev_0002"],
            }
        ],
        "overall_summary": "Strong statistical evidence for revenue growth.",
    })
    mock_llm.register("Lead Analyst", mock_analyst)

    mock_viz = json.dumps({
        "charts": [
            {
                "evidence_id": "ev_0002",
                "chart_type": "bar",
                "x": "label",
                "y": "value",
                "title": "Revenue Comparison",
            }
        ]
    })
    mock_llm.register("Visualization Agent", mock_viz)

    wf = DataPilotWorkflow(llm=mock_llm, store=store, snapshot_dir=snap_base)
    return wf, dataset_hash, store


def test_workflow_end_to_end(workflow_env):
    wf, dataset_hash, store = workflow_env

    initial_state: DataPilotState = {
        "run_id": "run-e2e-1",
        "dataset_hash": dataset_hash,
        "question": "Did revenue increase after the change?",
        "schema": {"sales": {"revenue_before": "DOUBLE", "revenue_after": "DOUBLE", "region": "VARCHAR"}},
        "role_map": {"revenue_before": "metric", "revenue_after": "metric", "region": "category"},
        "table_name": "sales",
        "snapshot_dir": str(wf.snapshot_dir),
        "evidence_ids": [],
        "evidences": [],
    }

    final_state = wf.run(initial_state)

    # Verify state progression
    assert "plan" in final_state
    assert len(final_state["plan"]["hypotheses"]) == 1
    assert "analysis" in final_state
    assert "evaluation" in final_state
    assert final_state["evaluation"]["overall_status"] in ("passed", "warning")
    assert "report" in final_state

    report = final_state["report"]
    assert report["firewall"]["passed"] is True
    assert "# DataPilot Investigation Report" in report["markdown"]
    assert "<!DOCTYPE html>" in report["html"]
    assert len(final_state["evidence_ids"]) >= 5

    # Verify ledger integrity
    assert store.verify_chain("run-e2e-1") is True
