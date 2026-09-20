import duckdb
from typing import Dict, Any, List
from datapilot.ledger.store import LedgerStore

class EDATools:
    def __init__(self, store: LedgerStore):
        self.store = store

    def correlation_matrix(self, run_id: str, dataset_hash: str, conn: duckdb.DuckDBPyConnection, 
                           table_name: str, role_map: Dict[str, str]) -> str:
        # Get numeric columns
        numeric_cols = []
        schema_info = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        for col in schema_info:
            cname = col[1]
            ctype = col[2].lower()
            if "int" in ctype or "float" in ctype or "double" in ctype or "decimal" in ctype:
                numeric_cols.append(cname)
                
        if len(numeric_cols) < 2:
            return self._record_ev(run_id, dataset_hash, "correlation_matrix", {}, {"matrix": {}}, numeric_cols)
            
        matrix = {}
        for c1 in numeric_cols:
            matrix[c1] = {}
            for c2 in numeric_cols:
                if c1 == c2:
                    matrix[c1][c2] = 1.0
                else:
                    corr = conn.execute(f"SELECT corr(\"{c1}\", \"{c2}\") FROM \"{table_name}\"").fetchone()[0]
                    matrix[c1][c2] = corr if corr is not None else 0.0
                    
        return self._record_ev(run_id, dataset_hash, "correlation_matrix", {}, {"matrix": matrix}, numeric_cols)

    def top_segments(self, run_id: str, dataset_hash: str, conn: duckdb.DuckDBPyConnection, 
                     table_name: str, role_map: Dict[str, str], top_n: int = 5) -> str:
        segments = [c for c, r in role_map.items() if r in ("segment", "region", "product", "channel")]
        metric = next((c for c, r in role_map.items() if r == "metric"), None)
        
        result = {}
        if not metric:
            return self._record_ev(run_id, dataset_hash, "top_segments", {"top_n": top_n}, result, segments)
            
        for seg in segments:
            query = f"""
                SELECT "{seg}", SUM("{metric}") as val
                FROM "{table_name}"
                WHERE "{seg}" IS NOT NULL
                GROUP BY "{seg}"
                ORDER BY val DESC
                LIMIT {top_n}
            """
            rows = conn.execute(query).fetchall()
            result[seg] = {str(r[0]): r[1] for r in rows}
            
        return self._record_ev(run_id, dataset_hash, "top_segments", {"top_n": top_n}, result, segments + [metric])

    def _record_ev(self, run_id, dataset_hash, code, params, result, columns) -> str:
        ev = self.store.record_evidence(
            run_id=run_id, kind="sql", produced_by="system", dataset_hash=dataset_hash,
            code=code, params=params, result=result, columns=columns, status="ok"
        )
        return ev.id
