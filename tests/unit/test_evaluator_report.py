"""
Tests for Phase 6: Evaluator Agent and Report Agent.
"""
import pytest
from datapilot.ledger.store import LedgerStore
from datapilot.agents.evaluator import EvaluatorAgent
from datapilot.agents.report_agent import ReportAgent
from datapilot.report.renderer import render_markdown, render_html


@pytest.fixture
def store(tmp_path):
    return LedgerStore(db_path=tmp_path / "ledger.db")


def test_evaluator_passing_evidence(store):
    evaluator = EvaluatorAgent(store=store)

    evidences = [
        {
            "id": "ev_0001",
            "kind": "stat_test",
            "status": "ok",
            "result": {
                "p": 0.001,
                "n": 100,
                "effect_size": 0.85,
            },
            "columns": ["rev_a", "rev_b"],
        },
        {
            "id": "ev_0002",
            "kind": "sql",
            "status": "ok",
            "result": {"num_rows": 25, "data": []},
            "columns": ["rev_a"],
        },
    ]

    res = evaluator.run(
        run_id="run-100",
        dataset_hash="hash123",
        evidences=evidences,
    )

    assert res["overall_status"] == "passed"
    assert res["overall_score"] >= 0.85
    assert len(res["issues"]) == 0
    assert "_evidence_id" in res
    ev = store.get_evidence(res["_evidence_id"])
    assert ev.kind == "evaluation"
    assert ev.produced_by == "evaluator"


def test_evaluator_flags_violations(store):
    evaluator = EvaluatorAgent(store=store)

    evidences = [
        {
            "id": "ev_0001",
            "kind": "stat_test",
            "status": "ok",
            "result": {
                "p": 0.25,      # Not significant
                "n": 15,        # Below minimum 30
                "vif_x": 8.5,   # High collinearity
                "effect_size": 0.05,  # Negligible
            },
            "columns": ["x", "y"],
        },
        {
            "id": "ev_0002",
            "kind": "model",
            "status": "ok",
            "result": {
                "metrics": {"r2": -0.2},
                "feature_importance": {"leaked_col": 0.99},
            },
            "columns": ["x", "y"],
        },
    ]

    res = evaluator.run(
        run_id="run-101",
        dataset_hash="hash123",
        evidences=evidences,
    )

    assert res["overall_status"] in ("warning", "failed")
    assert len(res["issues"]) >= 4
    # Issues should mention sample size, p-value, VIF, and leakage
    issues_str = " ".join(res["issues"]).lower()
    assert "sample size" in issues_str
    assert "significant" in issues_str
    assert "collinearity" in issues_str
    assert "leakage" in issues_str


def test_report_agent_and_renderer(store):
    agent = ReportAgent(store=store)

    plan = {
        "hypotheses": [{"id": "H1", "statement": "NA revenue is higher"}],
        "plan": [{"step": 1, "agent": "stats", "action": "before_after"}],
    }
    analysis = {
        "conclusions": [
            {
                "hypothesis_id": "H1",
                "verdict": "supported",
                "confidence": "high",
                "summary": "NA revenue was significantly higher with mean difference of 25.0.",
                "caveats": ["Assumes no external shocks."],
                "evidence_ids": ["ev_0001"],
            }
        ],
        "overall_summary": "Strong statistical evidence indicates revenue growth.",
    }
    evaluation = {
        "overall_status": "passed",
        "overall_score": 0.95,
        "issues": [],
    }
    charts = [
        {
            "type": "bar",
            "title": "Revenue by Region",
            "series": [{"label": "NA", "value": 150}, {"label": "EU", "value": 100}],
        }
    ]
    evidences = [
        {
            "id": "ev_0001",
            "kind": "stat_test",
            "produced_by": "stats_agent",
            "status": "ok",
            "result": {"p": 0.002, "n": 120, "diff": 25.0, "effect_size": 0.8},
            "columns": ["revenue", "region"],
            "hash": "abcdef1234567890",
        }
    ]

    report = agent.run(
        run_id="run-200",
        dataset_hash="hash999",
        question="Did NA revenue grow?",
        plan=plan,
        analysis=analysis,
        evaluation=evaluation,
        charts=charts,
        evidences=evidences,
    )

    assert report["firewall"]["passed"] is True
    assert "# DataPilot Investigation Report" in report["markdown"]
    assert "<!DOCTYPE html>" in report["html"]
    assert "NA revenue is higher" in report["markdown"]
    assert "_evidence_id" in report

    ev = store.get_evidence(report["_evidence_id"])
    assert ev.kind == "report"
    assert ev.status == "ok"
