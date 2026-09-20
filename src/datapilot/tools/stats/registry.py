from typing import Dict, Any, Callable
from datapilot.ledger.store import LedgerStore

class StatsRegistry:
    def __init__(self, store: LedgerStore):
        self.store = store
        self.tests: Dict[str, Callable] = {}
        
    def register(self, name: str, func: Callable):
        self.tests[name] = func
        
    def execute(self, run_id: str, dataset_hash: str, test_name: str, 
                args: Dict[str, Any], columns: list[str]) -> str:
        if test_name not in self.tests:
            ev = self.store.record_evidence(
                run_id=run_id, kind="stat_test", produced_by="stats", dataset_hash=dataset_hash,
                code=test_name, params=args, result={}, columns=columns, status="error",
                error=f"Test {test_name} not found in registry"
            )
            return ev.id
            
        try:
            result = self.tests[test_name](**args)
            ev = self.store.record_evidence(
                run_id=run_id, kind="stat_test", produced_by="stats", dataset_hash=dataset_hash,
                code=test_name, params=args, result=result, columns=columns, status="ok"
            )
            return ev.id
        except Exception as e:
            ev = self.store.record_evidence(
                run_id=run_id, kind="stat_test", produced_by="stats", dataset_hash=dataset_hash,
                code=test_name, params=args, result={}, columns=columns, status="error",
                error=str(e)
            )
            return ev.id
