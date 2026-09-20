"""
Planner Agent — given a question + schema, produces hypotheses and a step plan.
"""
import json
from typing import Dict, Any, List
from datapilot.llm.client import LLMClient, LLMMessage
from datapilot.prompts.templates import PLANNER_SYSTEM
from datapilot.ledger.store import LedgerStore


class PlannerAgent:
    def __init__(self, llm: LLMClient, store: LedgerStore):
        self.llm = llm
        self.store = store

    def run(
        self,
        run_id: str,
        dataset_hash: str,
        question: str,
        schema: Dict[str, Any],
        role_map: Dict[str, str],
    ) -> Dict[str, Any]:
        schema_text = json.dumps(schema, indent=2)
        roles_text = json.dumps(role_map, indent=2)
        user_msg = LLMMessage(
            "user",
            f"Question: {question}\n\nSchema:\n{schema_text}\n\nColumn roles:\n{roles_text}",
        )
        plan = self.llm.complete_json(PLANNER_SYSTEM, [user_msg])

        ev = self.store.record_evidence(
            run_id=run_id,
            kind="plan",
            produced_by="planner",
            dataset_hash=dataset_hash,
            code="plan",
            params={"question": question},
            result=plan,
            columns=list(role_map.keys()),
            status="ok",
        )
        plan["_evidence_id"] = ev.id
        return plan
