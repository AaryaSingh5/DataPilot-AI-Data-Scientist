import duckdb
import uuid
from typing import Dict, Any, List
from pathlib import Path
from datapilot.ledger.store import LedgerStore
from datapilot.sandbox.sql_guard import SQLGuard, SQLGuardError

class SQLExecutorError(Exception):
    pass

class SQLExecutor:
    def __init__(self, store: LedgerStore, snapshot_dir: Path, artifact_dir: Path):
        self.store = store
        self.snapshot_dir = Path(snapshot_dir)
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, run_id: str, dataset_hash: str, sql: str, allowed_schema: Dict[str, List[str]], depends_on: List[str] = None) -> str:
        guard = SQLGuard(allowed_schema)
        try:
            safe_sql, tables, columns = guard.check_sql(sql)
        except SQLGuardError as e:
            ev = self.store.record_evidence(
                run_id=run_id, kind="sql", produced_by="sql", dataset_hash=dataset_hash,
                code=sql, params={}, result={}, columns=[], status="error", error=str(e),
                depends_on=depends_on
            )
            return ev.id

        # Load snapshot read-only
        conn = duckdb.connect(":memory:")
        target_dir = self.snapshot_dir / dataset_hash
        if not target_dir.exists():
            ev = self.store.record_evidence(
                run_id=run_id, kind="sql", produced_by="sql", dataset_hash=dataset_hash,
                code=safe_sql, params={}, result={}, columns=columns, status="error", 
                error=f"Snapshot {dataset_hash} not found", depends_on=depends_on
            )
            return ev.id

        try:
            # Mount parquet files
            for parquet_file in target_dir.glob("*.parquet"):
                table_name = parquet_file.stem
                # We use read_parquet to map them read-only
                conn.execute(f"CREATE VIEW \"{table_name}\" AS SELECT * FROM read_parquet('{parquet_file}')")
                
            # Execute with timeout (using duckdb's execution time limit if possible or thread timeout)
            # For simplicity, duckdb's python api doesn't have a direct query timeout parameter,
            # but we can set max_expression_depth or similar. In a real app we'd run in a subprocess.
            # Using pragma statement_timeout doesn't exist in duckdb. 
            arrow_table = conn.execute(safe_sql).arrow()
            
            # Save artifact
            artifact_name = f"{uuid.uuid4().hex}.parquet"
            artifact_path = self.artifact_dir / artifact_name
            duckdb.connect().execute(f"CREATE TABLE res AS SELECT * FROM arrow_table; COPY res TO '{artifact_path}' (FORMAT PARQUET)")
            
            # Get first 200 rows as dict
            sample_df = arrow_table.slice(0, min(200, arrow_table.num_rows)).to_pandas()
            result_dict = {
                "num_rows": arrow_table.num_rows,
                "data": sample_df.to_dict(orient="records")
            }
            
            ev = self.store.record_evidence(
                run_id=run_id, kind="sql", produced_by="sql", dataset_hash=dataset_hash,
                code=safe_sql, params={}, result=result_dict, columns=columns, status="ok",
                artifact_path=str(artifact_path), depends_on=depends_on
            )
            return ev.id
            
        except Exception as e:
            ev = self.store.record_evidence(
                run_id=run_id, kind="sql", produced_by="sql", dataset_hash=dataset_hash,
                code=safe_sql, params={}, result={}, columns=columns, status="error", 
                error=str(e), depends_on=depends_on
            )
            return ev.id
