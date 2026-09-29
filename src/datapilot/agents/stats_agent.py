"""
Stats Agent — selects and runs the appropriate statistical test based on a hypothesis.
"""
import json
from typing import Dict, Any, List
import duckdb

from datapilot.llm.client import LLMClient, LLMMessage
from datapilot.prompts.templates import STATS_AGENT_SYSTEM
from datapilot.ledger.store import LedgerStore
from datapilot.tools.stats.comparisons import before_after
from datapilot.tools.stats.association import association, controlled_association
from datapilot.tools.stats.timeseries import analyze_timeseries
from datapilot.tools.stats.power import power_analysis
from datapilot.tools.stats.correction import apply_fdr_correction


class StatsAgent:
    def __init__(self, llm: LLMClient, store: LedgerStore):
        self.llm = llm
        self.store = store

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
        test_type = hypothesis.get("test_type", "")

        # Ask LLM which test to run and which column maps to which param
        schema = {col: {} for col in columns}
        user_msg = LLMMessage(
            "user",
            f"Hypothesis {hypothesis_id}: {statement}\n\nTest type hint: {test_type}\nColumns: {json.dumps(columns)}",
        )
        config = self.llm.complete_json(STATS_AGENT_SYSTEM, [user_msg])
        test_name = config.get("test", test_type)
        col_map = config.get("columns", {})

        try:
            result, py_code = self._dispatch(test_name, col_map, conn, table_name, columns)
            status = "ok"
            err = None
        except Exception as e:
            result = {}
            py_code = f"# Failed to run {test_name}\n# {str(e)}"
            status = "error"
            err = str(e)

        purpose = f"Run {test_name} to test hypothesis: {statement}"
        
        ev = self.store.record_evidence(
            run_id=run_id,
            kind="stat_test",
            produced_by="stats_agent",
            dataset_hash=dataset_hash,
            code=py_code,
            params={"hypothesis_id": hypothesis_id, "col_map": col_map},
            result=result,
            columns=columns,
            status=status,
            error=err,
            depends_on=depends_on,
            purpose=purpose,
        )
        return ev.id

    def _dispatch(self, test_name: str, col_map: Dict, conn, table_name: str, columns: List[str]) -> tuple[Dict, str]:
        def fetch(col):
            rows = conn.execute(f'SELECT "{col}" FROM "{table_name}" WHERE "{col}" IS NOT NULL').fetchall()
            return [r[0] for r in rows]

        if test_name == "before_after":
            a_col = col_map.get("a", columns[0])
            b_col = col_map.get("b", columns[1] if len(columns) > 1 else columns[0])
            code = f"""import pingouin as pg
import scipy.stats as stats
import numpy as np

# before_after test
a = df['{a_col}'].dropna().values
b = df['{b_col}'].dropna().values
_, p_var = stats.levene(a, b)
equal_var = p_var >= 0.05
res = pg.ttest(b, a, correction=not equal_var)
print(res)
"""
            return before_after(fetch(a_col), fetch(b_col)), code

        elif test_name == "association":
            x_col = col_map.get("x", columns[0])
            y_col = col_map.get("y", columns[1] if len(columns) > 1 else columns[0])
            code = f"""import pingouin as pg

# association test
x = df['{x_col}'].dropna().values
y = df['{y_col}'].dropna().values
res = pg.corr(x, y)
print(res)
"""
            return association(fetch(x_col), fetch(y_col)), code

        elif test_name == "controlled_association":
            x_col = col_map.get("x", columns[0])
            y_col = col_map.get("y", columns[1])
            control_cols = [c for c in columns if c not in (x_col, y_col)]
            controls = {c: fetch(c) for c in control_cols}
            ctrl_str = ", ".join([f"'{c}'" for c in control_cols])
            code = f"""import pingouin as pg

# controlled_association test
covar = [{ctrl_str}]
clean_df = df[['{x_col}', '{y_col}'] + covar].dropna()
res = pg.partial_corr(data=clean_df, x='{x_col}', y='{y_col}', covar=covar)
print(res)
"""
            return controlled_association(fetch(x_col), fetch(y_col), controls), code

        elif test_name == "timeseries":
            val_col = col_map.get("values", columns[0])
            code = f"""import pandas as pd
import statsmodels.api as sm

# timeseries test
y = df['{val_col}'].dropna().values
x = range(len(y))
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())
"""
            return analyze_timeseries(fetch(val_col)), code

        else:
            raise ValueError(f"Unknown test: {test_name}")
