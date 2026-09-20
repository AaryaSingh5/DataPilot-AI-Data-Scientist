import math
from typing import Dict, Any, List
from pydantic import BaseModel
from datapilot.ledger.store import LedgerStore
from datapilot.ledger.provenance import ProvenanceDAG

class ReplayReport(BaseModel):
    evidence_id: str
    match: bool
    diffs: Dict[str, Any]

class ReplayEngine:
    def __init__(self, store: LedgerStore):
        self.store = store

    def _compare_values(self, v1: Any, v2: Any, rtol: float = 1e-9) -> bool:
        if isinstance(v1, float) and isinstance(v2, float):
            if math.isnan(v1) and math.isnan(v2):
                return True
            if math.isinf(v1) and math.isinf(v2):
                return v1 == v2
            return math.isclose(v1, v2, rel_tol=rtol)
        return v1 == v2

    def _compare_dicts(self, d1: dict, d2: dict) -> Dict[str, Any]:
        diffs = {}
        all_keys = set(d1.keys()).union(set(d2.keys()))
        for key in all_keys:
            if key not in d1:
                diffs[key] = {"expected": None, "actual": d2[key], "reason": "missing_in_expected"}
            elif key not in d2:
                diffs[key] = {"expected": d1[key], "actual": None, "reason": "missing_in_actual"}
            else:
                if not self._compare_values(d1[key], d2[key]):
                    diffs[key] = {"expected": d1[key], "actual": d2[key], "reason": "mismatch"}
        return diffs

    def replay(self, evidence_id: str) -> ReplayReport:
        # Skeleton:
        # 1. Fetch ev_id from store
        # 2. Reload snapshot by dataset_hash
        # 3. Route to the right tool based on kind/produced_by, using ev.code and ev.params
        # 4. Compare resulting dictionary with ev.result using _compare_dicts
        
        # Currently a stub for Phase 1
        return ReplayReport(evidence_id=evidence_id, match=True, diffs={})

    def replay_run(self, run_id: str) -> List[ReplayReport]:
        dag = ProvenanceDAG(self.store, run_id)
        # Process in topological order
        sorted_evs = list(nx.topological_sort(dag.to_networkx()))
        reports = []
        for ev_id in sorted_evs:
            reports.append(self.replay(ev_id))
        return reports
