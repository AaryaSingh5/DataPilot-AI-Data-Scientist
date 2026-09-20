"""
SQL Agent — generates, guards, and executes a single DuckDB SELECT query.
"""
import json
from typing import Dict, Any, List
from datapilot.llm.client import LLMClient, LLMMessage
from datapilot.prompts.templates import SQL_AGENT_SYSTEM
from datapilot.sandbox.sql_guard import SQLGuard, SQLGuardError
from datapilot.ledger.store import LedgerStore
from pathlib import Path
import duckdb


class SQLAgent:
    def __init__(self, llm: LLMClient, store: LedgerStore, snapshot_dir: Path):
        self.llm = llm
        self.store = store
        self.snapshot_dir = Path(snapshot_dir)

    def run(
        self,
        run_id: str,
        dataset_hash: str,
        question: str,
        schema: Dict[str, Any],
        role_map: Dict[str, str],
        depends_on: List[str] = None,
        hypothesis_id: str = "",
    ) -> str:
        schema_text = json.dumps(schema, indent=2)
        user_msg = LLMMessage(
            "user",
            f"Hypothesis {hypothesis_id}: {question}\n\nSchema:\n{schema_text}",
        )
        parsed = self.llm.complete_json(SQL_AGENT_SYSTEM, [user_msg])
        raw_sql = parsed.get("sql", "")

        guard = SQLGuard(schema)
        try:
            safe_sql, tables, cols = guard.check_sql(raw_sql)
        except SQLGuardError as e:
            ev = self.store.record_evidence(
                run_id=run_id, kind="sql", produced_by="sql_agent",
                dataset_hash=dataset_hash, code=raw_sql,
                params={"question": question}, result={}, columns=[],
                status="error", error=str(e), depends_on=depends_on,
            )
            return ev.id

        # Execute against snapshot
        snap_dir = self.snapshot_dir / dataset_hash
        conn = duckdb.connect(":memory:")
        for pq in snap_dir.glob("*.parquet"):
            conn.execute(f"CREATE VIEW \"{pq.stem}\" AS SELECT * FROM read_parquet('{pq}')")

        try:
            rows = conn.execute(safe_sql).fetchall()
            col_names = [d[0] for d in conn.description]
            data = [dict(zip(col_names, r)) for r in rows]
            result = {"num_rows": len(data), "data": data[:200]}
            status = "ok"
            err = None
        except Exception as exc:
            result = {}
            status = "error"
            err = str(exc)

        ev = self.store.record_evidence(
            run_id=run_id, kind="sql", produced_by="sql_agent",
            dataset_hash=dataset_hash, code=safe_sql,
            params={"question": question, "hypothesis_id": hypothesis_id},
            result=result, columns=cols, status=status, error=err,
            depends_on=depends_on,
        )
        return ev.id
