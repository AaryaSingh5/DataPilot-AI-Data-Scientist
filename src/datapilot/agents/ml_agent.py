"""
ML Agent — selects target/task and runs gradient-boosting with feature importance.
"""
import json
from typing import Dict, Any, List
from pathlib import Path
import duckdb
import pandas as pd

from datapilot.llm.client import LLMClient, LLMMessage
from datapilot.prompts.templates import ML_AGENT_SYSTEM
from datapilot.ledger.store import LedgerStore
from datapilot.tools.ml import train_gradient_boosting


class MLAgent:
    def __init__(self, llm: LLMClient, store: LedgerStore, snapshot_dir: Path):
        self.llm = llm
        self.store = store
        self.snapshot_dir = Path(snapshot_dir)

    def run(
        self,
        run_id: str,
        dataset_hash: str,
        hypothesis: Dict[str, Any],
        conn: duckdb.DuckDBPyConnection,
        table_name: str,
        depends_on: List[str] = None,
    ) -> str:
        hypothesis_id = hypothesis.get("id", "")
        statement = hypothesis.get("statement", "")
        columns = hypothesis.get("columns", [])

        user_msg = LLMMessage(
            "user",
            f"Hypothesis {hypothesis_id}: {statement}\nAvailable columns: {json.dumps(columns)}",
        )
        config = self.llm.complete_json(ML_AGENT_SYSTEM, [user_msg])
        target = config.get("target", columns[-1] if columns else "")
        task = config.get("task", "regression")
        drop_cols = config.get("drop_columns", [])

        # Load data
        snap_dir = self.snapshot_dir / dataset_hash
        pq_files = list(snap_dir.glob("*.parquet"))
        if not pq_files:
            ev = self.store.record_evidence(
                run_id=run_id, kind="model", produced_by="ml_agent",
                dataset_hash=dataset_hash, code="gradient_boosting",
                params=config, result={}, columns=columns,
                status="error", error="No snapshot parquet found",
                depends_on=depends_on,
            )
            return ev.id

        df = pd.read_parquet(pq_files[0])
        if drop_cols:
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        try:
            result = train_gradient_boosting(df, target_col=target, task=task)
            status = "ok"
            err = None
        except Exception as e:
            result = {}
            status = "error"
            err = str(e)

        ev = self.store.record_evidence(
            run_id=run_id, kind="model", produced_by="ml_agent",
            dataset_hash=dataset_hash, code="gradient_boosting",
            params={**config, "hypothesis_id": hypothesis_id},
            result=result, columns=columns, status=status, error=err,
            depends_on=depends_on,
        )
        return ev.id
