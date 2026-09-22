"""
Multi-turn Follow-up Session Manager — maintains context, links evidence across turns in the DAG.
"""
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path

from datapilot.graph.workflow import DataPilotWorkflow
from datapilot.graph.state import DataPilotState
from datapilot.ledger.store import LedgerStore
from datapilot.llm.client import LLMClient


class SessionTurn:
    def __init__(
        self,
        turn_index: int,
        run_id: str,
        question: str,
        state: DataPilotState,
    ):
        self.turn_index = turn_index
        self.run_id = run_id
        self.question = question
        self.state = state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "run_id": self.run_id,
            "question": self.question,
            "report": self.state.get("report", {}),
            "evidence_ids": self.state.get("evidence_ids", []),
            "firewall_passed": self.state.get("firewall_passed", False),
        }


class SessionManager:
    def __init__(
        self,
        workflow: DataPilotWorkflow,
        dataset_hash: str,
        schema: Dict[str, Any],
        role_map: Dict[str, str],
        session_id: Optional[str] = None,
        table_name: Optional[str] = None,
    ):
        self.workflow = workflow
        self.dataset_hash = dataset_hash
        self.schema = schema
        self.role_map = role_map
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.table_name = table_name or ""
        self.turns: List[SessionTurn] = []
        self.all_evidence_ids: List[str] = []

    def ask(self, question: str) -> Dict[str, Any]:
        turn_index = len(self.turns) + 1
        run_id = f"{self.session_id}_t{turn_index}"

        # Contextual question for follow-ups
        effective_question = question
        if self.turns:
            prev_context = []
            for t in self.turns:
                summ = t.state.get("report", {}).get("executive_summary", "")
                prev_context.append(f"Q: {t.question} -> Finding: {summ[:100]}")
            context_str = "\n".join(prev_context)
            effective_question = f"Previous findings in this session:\n{context_str}\n\nFollow-up Question: {question}"

        initial_state: DataPilotState = {
            "run_id": run_id,
            "dataset_hash": self.dataset_hash,
            "question": effective_question,
            "schema": self.schema,
            "role_map": self.role_map,
            "table_name": self.table_name,
            "snapshot_dir": str(self.workflow.snapshot_dir),
            "evidence_ids": list(self.all_evidence_ids),
            "evidences": [],
        }

        final_state = self.workflow.run(initial_state)

        turn = SessionTurn(
            turn_index=turn_index,
            run_id=run_id,
            question=question,
            state=final_state,
        )
        self.turns.append(turn)

        # Accumulate evidence IDs across turns
        new_eids = final_state.get("evidence_ids", [])
        for eid in new_eids:
            if eid not in self.all_evidence_ids:
                self.all_evidence_ids.append(eid)

        return turn.to_dict()

    def get_session_history(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.turns]
