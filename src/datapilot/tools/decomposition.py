import duckdb
from typing import Dict, Any, List
from datapilot.ledger.store import LedgerStore

class DecompositionError(Exception):
    pass

class DecompositionTools:
    def __init__(self, store: LedgerStore):
        self.store = store

    def revenue_decomposition(self, run_id: str, dataset_hash: str, conn: duckdb.DuckDBPyConnection, 
                              table_name: str, role_map: Dict[str, str], 
                              period_a_filter: str, period_b_filter: str) -> str:
        
        customer_id = next((c for c, r in role_map.items() if r == "customer_id"), None)
        metric = next((c for c, r in role_map.items() if r == "metric"), None)
        
        if not customer_id or not metric:
            raise DecompositionError("Missing customer_id or metric role")

        def get_stats(filt):
            query = f"""
                SELECT 
                    COUNT(DISTINCT "{customer_id}") as customers,
                    SUM("{metric}") as total_rev
                FROM "{table_name}"
                WHERE {filt}
            """
            c, r = conn.execute(query).fetchone()
            c = c or 0
            r = r or 0
            arpu = r / c if c > 0 else 0
            return c, r, arpu

        c_a, r_a, arpu_a = get_stats(period_a_filter)
        c_b, r_b, arpu_b = get_stats(period_b_filter)
        
        delta_r = r_b - r_a
        delta_c = c_b - c_a
        delta_arpu = arpu_b - arpu_a
        
        # Exact decomposition: DeltaR = DeltaC * ARPU_A + DeltaARPU * C_A + DeltaC * DeltaARPU
        c_effect = delta_c * arpu_a
        arpu_effect = delta_arpu * c_a
        interaction = delta_c * delta_arpu
        
        reconciled = c_effect + arpu_effect + interaction
        
        if abs(reconciled - delta_r) > 1e-5:
            raise DecompositionError("Decomposition failed to reconcile with total change")
            
        result = {
            "period_a": {"customers": c_a, "revenue": r_a, "arpu": arpu_a},
            "period_b": {"customers": c_b, "revenue": r_b, "arpu": arpu_b},
            "delta_revenue": delta_r,
            "contribution_customers": c_effect,
            "contribution_arpu": arpu_effect,
            "contribution_interaction": interaction
        }
        
        ev = self.store.record_evidence(
            run_id=run_id, kind="sql", produced_by="system", dataset_hash=dataset_hash,
            code="revenue_decomposition", params={"filter_a": period_a_filter, "filter_b": period_b_filter},
            result=result, columns=[customer_id, metric], status="ok"
        )
        return ev.id
