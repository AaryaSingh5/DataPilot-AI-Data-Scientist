"""
Analyst Agent — synthesizes all evidence into conclusions with verdicts and confidence.
"""
import json
from typing import Dict, Any, List

from datapilot.llm.client import LLMClient, LLMMessage
from datapilot.prompts.templates import ANALYST_SYSTEM
from datapilot.ledger.store import LedgerStore
from datapilot.tools.stats.correction import apply_fdr_correction


class AnalystAgent:
    def __init__(self, llm: LLMClient, store: LedgerStore):
        self.llm = llm
        self.store = store

    def run(
        self,
        run_id: str,
        dataset_hash: str,
        plan: Dict[str, Any],
        evidences: List[Dict[str, Any]],
        depends_on: List[str] = None,
    ) -> str:
        # Apply FDR correction across all stat tests before synthesis
        stat_evidences = [e for e in evidences if e.get("kind") == "stat_test"]
        apply_fdr_correction(stat_evidences)

        ev_summary = json.dumps(
            [{
                "id": e["id"],
                "kind": e["kind"],
                "status": e.get("status"),
                "result": e.get("result", {}),
                "columns": e.get("columns", []),
                "error": e.get("error"),
            } for e in evidences],
            indent=2,
            default=str,
        )
        hypotheses_text = json.dumps(plan.get("hypotheses", []), indent=2)

        user_msg = LLMMessage(
            "user",
            f"Hypotheses:\n{hypotheses_text}\n\nEvidence:\n{ev_summary}",
        )
        analysis = self.llm.complete_json(ANALYST_SYSTEM, [user_msg])

        ev = self.store.record_evidence(
            run_id=run_id,
            kind="analysis",
            produced_by="analyst",
            dataset_hash=dataset_hash,
            code="synthesis",
            params={},
            result=analysis,
            columns=[],
            status="ok",
            depends_on=depends_on,
        )
        return ev.id
