"""Phase 5 agent tests using MockLLMClient."""
import json
import pytest
import duckdb
from pathlib import Path
from datapilot.llm.client import MockLLMClient
from datapilot.ledger.store import LedgerStore
from datapilot.agents.planner import PlannerAgent
from datapilot.agents.analyst import AnalystAgent
from datapilot.agents.sql_agent import SQLAgent
from datapilot.agents.stats_agent import StatsAgent
from datapilot.agents.ml_agent import MLAgent
from datapilot.agents.viz_agent import VizAgent


MOCK_PLAN = json.dumps({
    "hypotheses": [
        {
            "id": "H1",
            "statement": "Revenue is higher in region NA than EU",
            "columns": ["revenue", "region"],
            "test_type": "before_after",
        }
    ],
    "plan": [
        {"step": 1, "agent": "stats", "action": "Run before_after on revenue by region", "hypothesis_id": "H1"}
    ],
})

MOCK_ANALYSIS = json.dumps({
    "conclusions": [
        {
            "hypothesis_id": "H1",
            "verdict": "supported",
            "confidence": "high",
            "summary": "Revenue is significantly higher in NA.",
            "caveats": [],
            "evidence_ids": ["ev-001"],
        }
    ],
    "overall_summary": "NA region drives most revenue.",
})


@pytest.fixture
def store(tmp_path):
    return LedgerStore(db_path=tmp_path / "ledger.db")


@pytest.fixture
def mock_llm():
    llm = MockLLMClient()
    llm.register("PLANNER_QUESTION", MOCK_PLAN)
    # Analyst system prompt contains "Lead Analyst" — use that as trigger
    llm.register("Lead Analyst", MOCK_ANALYSIS)
    return llm


def test_planner_returns_plan(store, mock_llm):
    agent = PlannerAgent(llm=mock_llm, store=store)
    schema = {"revenue": {}, "region": {}}
    role_map = {"revenue": "metric", "region": "region"}

    plan = agent.run(
        run_id="run-1",
        dataset_hash="abc123",
        question="PLANNER_QUESTION: Is revenue higher in NA than EU?",
        schema=schema,
        role_map=role_map,
    )

    assert "hypotheses" in plan
    assert len(plan["hypotheses"]) == 1
    assert plan["hypotheses"][0]["id"] == "H1"
    assert "_evidence_id" in plan


def test_analyst_returns_conclusions(store, mock_llm):
    agent = AnalystAgent(llm=mock_llm, store=store)

    fake_evidences = [
        {
            "id": "ev-001",
            "kind": "stat_test",
            "status": "ok",
            "result": {"p": 0.002, "effect_size": 0.8},
            "columns": ["revenue"],
            "error": None,
        }
    ]
    plan = {"hypotheses": [{"id": "H1", "statement": "Revenue is higher in NA"}]}

    ev_id = agent.run(
        run_id="run-1",
        dataset_hash="abc123",
        plan=plan,
        evidences=fake_evidences,
    )
    assert ev_id.startswith("ev_")


def test_sql_agent(tmp_path, store):
    mock_llm = MockLLMClient()
    mock_llm.register("SQL Agent", json.dumps({"sql": 'SELECT region, SUM(revenue) as rev FROM "sales" GROUP BY region LIMIT 10'}))

    # Setup snapshot dir with parquet
    snap_dir = tmp_path / "snapshots" / "abc123"
    snap_dir.mkdir(parents=True)
    import pandas as pd
    df = pd.DataFrame({"region": ["NA", "EU", "NA"], "revenue": [100, 200, 150]})
    df.to_parquet(snap_dir / "sales.parquet")

    agent = SQLAgent(llm=mock_llm, store=store, snapshot_dir=tmp_path / "snapshots")
    schema = {"sales": {"region": "VARCHAR", "revenue": "BIGINT"}}
    role_map = {"region": "category", "revenue": "metric"}

    ev_id = agent.run(
        run_id="run-1",
        dataset_hash="abc123",
        question="What is total revenue by region?",
        schema=schema,
        role_map=role_map,
    )
    assert ev_id.startswith("ev_")
    ev = store.get_evidence(ev_id)
    assert ev.status == "ok"
    assert len(ev.result["data"]) == 2


def test_stats_agent(store):
    mock_llm = MockLLMClient()
    mock_llm.register("Stats Agent", json.dumps({
        "test": "before_after",
        "columns": {"a": "rev_before", "b": "rev_after"},
        "justification": "Compare revenue before and after",
    }))

    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE sales (rev_before DOUBLE, rev_after DOUBLE)")
    for a, b in zip([10.0, 12.0, 11.0, 13.0, 12.0], [20.0, 22.0, 21.0, 23.0, 22.0]):
        conn.execute(f"INSERT INTO sales VALUES ({a}, {b})")

    agent = StatsAgent(llm=mock_llm, store=store)
    hypothesis = {
        "id": "H1",
        "statement": "Revenue is higher after change",
        "columns": ["rev_before", "rev_after"],
        "test_type": "before_after",
    }

    ev_id = agent.run(
        run_id="run-1",
        dataset_hash="abc123",
        hypothesis=hypothesis,
        conn=conn,
        table_name="sales",
    )
    assert ev_id.startswith("ev_")
    ev = store.get_evidence(ev_id)
    assert ev.status == "ok"
    assert "p" in ev.result


def test_ml_agent(tmp_path, store):
    mock_llm = MockLLMClient()
    mock_llm.register("ML Agent", json.dumps({
        "target": "target",
        "task": "regression",
        "drop_columns": [],
        "rationale": "Predict target from features",
    }))

    snap_dir = tmp_path / "snapshots" / "abc123"
    snap_dir.mkdir(parents=True)
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    df = pd.DataFrame({
        "f1": np.random.randn(50),
        "f2": np.random.randn(50),
        "target": np.random.randn(50),
    })
    df.to_parquet(snap_dir / "data.parquet")

    agent = MLAgent(llm=mock_llm, store=store, snapshot_dir=tmp_path / "snapshots")
    conn = duckdb.connect(":memory:")
    hypothesis = {
        "id": "H2",
        "statement": "Features predict target",
        "columns": ["f1", "f2", "target"],
    }

    ev_id = agent.run(
        run_id="run-1",
        dataset_hash="abc123",
        hypothesis=hypothesis,
        conn=conn,
        table_name="data",
    )
    assert ev_id.startswith("ev_")
    ev = store.get_evidence(ev_id)
    assert ev.status == "ok"
    assert "metrics" in ev.result or "r2" in str(ev.result) or "cv_scores" in ev.result


def test_viz_agent(store):
    mock_llm = MockLLMClient()
    mock_llm.register("Visualization Agent", json.dumps({
        "charts": [{
            "evidence_id": "ev_0001",
            "chart_type": "bar",
            "x": "region",
            "y": "revenue",
            "title": "Revenue by Region",
        }]
    }))

    agent = VizAgent(llm=mock_llm, store=store)
    evidences = [
        {
            "id": "ev_0001",
            "kind": "sql",
            "result": {
                "data": [
                    {"region": "NA", "revenue": 100},
                    {"region": "EU", "revenue": 200},
                ]
            },
            "columns": ["region", "revenue"],
        }
    ]

    ev_id = agent.run(
        run_id="run-1",
        dataset_hash="abc123",
        evidences=evidences,
    )
    assert ev_id.startswith("ev_")
    ev = store.get_evidence(ev_id)
    assert ev.status == "ok"
    assert len(ev.result["charts"]) == 1
    assert ev.result["charts"][0]["type"] == "bar"
