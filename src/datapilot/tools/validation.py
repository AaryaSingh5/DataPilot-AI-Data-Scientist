import duckdb
from typing import Dict, Any, List
from datapilot.ledger.store import LedgerStore

class ValidationChecks:
    def __init__(self, store: LedgerStore, thresholds: Dict[str, Any]):
        self.store = store
        self.thresholds = thresholds

    def run_all(self, run_id: str, dataset_hash: str, conn: duckdb.DuckDBPyConnection, 
                table_name: str, role_map: Dict[str, str]) -> List[str]:
        # Return list of evidence IDs generated
        ev_ids = []
        ev_ids.append(self.check_nulls(run_id, dataset_hash, conn, table_name, role_map))
        ev_ids.append(self.check_duplicates(run_id, dataset_hash, conn, table_name, role_map))
        ev_ids.append(self.check_negative_metrics(run_id, dataset_hash, conn, table_name, role_map))
        return ev_ids

    def check_nulls(self, run_id: str, dataset_hash: str, conn: duckdb.DuckDBPyConnection, 
                    table_name: str, role_map: Dict[str, str]) -> str:
        max_null_rate = self.thresholds.get("validation", {}).get("max_null_rate_metric", 0.02)
        
        target_cols = [col for col, role in role_map.items() if role in ("metric", "date", "quantity", "price")]
        if not target_cols:
            return self._record_ev(run_id, dataset_hash, "check_nulls", {}, {"status": "skipped", "severity": "info"}, [])

        row_count = conn.execute(f"SELECT COUNT(*) FROM \"{table_name}\"").fetchone()[0]
        if row_count == 0:
            return self._record_ev(run_id, dataset_hash, "check_nulls", {}, {"status": "skipped", "severity": "info"}, target_cols)

        issues = {}
        severity = "info"
        
        for col in target_cols:
            non_null = conn.execute(f"SELECT COUNT(\"{col}\") FROM \"{table_name}\"").fetchone()[0]
            null_rate = (row_count - non_null) / row_count
            if null_rate > 0:
                issues[col] = null_rate
                if null_rate > max_null_rate:
                    severity = "blocking"
                elif severity != "blocking":
                    severity = "warning"
                    
        return self._record_ev(run_id, dataset_hash, "check_nulls", {"max_null_rate": max_null_rate}, 
                               {"severity": severity, "issues": issues}, target_cols)

    def check_duplicates(self, run_id: str, dataset_hash: str, conn: duckdb.DuckDBPyConnection, 
                         table_name: str, role_map: Dict[str, str]) -> str:
        # Simplistic check for exact duplicate rows
        row_count = conn.execute(f"SELECT COUNT(*) FROM \"{table_name}\"").fetchone()[0]
        distinct_count = conn.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM \"{table_name}\")").fetchone()[0]
        
        duplicates = row_count - distinct_count
        severity = "warning" if duplicates > 0 else "info"
        
        return self._record_ev(run_id, dataset_hash, "check_duplicates", {}, 
                               {"severity": severity, "duplicates": duplicates}, [])

    def check_negative_metrics(self, run_id: str, dataset_hash: str, conn: duckdb.DuckDBPyConnection, 
                               table_name: str, role_map: Dict[str, str]) -> str:
        metrics = [col for col, role in role_map.items() if role in ("metric", "price", "quantity")]
        issues = {}
        severity = "info"
        
        for col in metrics:
            # Check if column is numeric
            col_type = conn.execute(f"SELECT typeof(\"{col}\") FROM \"{table_name}\" LIMIT 1").fetchone()[0].lower()
            if "int" in col_type or "float" in col_type or "double" in col_type or "decimal" in col_type:
                neg_count = conn.execute(f"SELECT COUNT(*) FROM \"{table_name}\" WHERE \"{col}\" < 0").fetchone()[0]
                if neg_count > 0:
                    issues[col] = neg_count
                    severity = "warning"
                    
        return self._record_ev(run_id, dataset_hash, "check_negative_metrics", {}, 
                               {"severity": severity, "issues": issues}, metrics)

    def _record_ev(self, run_id, dataset_hash, check_name, params, result, columns) -> str:
        ev = self.store.record_evidence(
            run_id=run_id, kind="validation", produced_by="system", dataset_hash=dataset_hash,
            code=check_name, params=params, result=result, columns=columns, status="ok"
        )
        return ev.id
