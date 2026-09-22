"""
Viz Agent — selects chart type and builds chart specs for each evidence item.
"""
import json
from typing import Dict, Any, List

from datapilot.llm.client import LLMClient, LLMMessage
from datapilot.prompts.templates import VIZ_AGENT_SYSTEM
from datapilot.ledger.store import LedgerStore
from datapilot.tools import viz as viz_tools


class VizAgent:
    def __init__(self, llm: LLMClient, store: LedgerStore):
        self.llm = llm
        self.store = store

    def run(
        self,
        run_id: str,
        dataset_hash: str,
        evidences: List[Dict[str, Any]],
        depends_on: List[str] = None,
    ) -> str:
        """
        Given a list of evidence dicts (id, kind, result, columns),
        ask LLM to pick chart types and build specs.
        Returns an evidence ID for the chart collection.
        """
        ev_summary = json.dumps(
            [{"id": e["id"], "kind": e["kind"], "columns": e.get("columns", []),
              "result_keys": list(e.get("result", {}).keys())} for e in evidences],
            indent=2,
        )
        user_msg = LLMMessage("user", f"Evidence items:\n{ev_summary}")
        chart_config = self.llm.complete_json(VIZ_AGENT_SYSTEM, [user_msg])

        charts = []
        for chart_def in chart_config.get("charts", []):
            ev_id = chart_def.get("evidence_id", "")
            chart_type = chart_def.get("chart_type", "bar")
            title = chart_def.get("title", "")

            # Find the matching evidence
            ev_data = next((e for e in evidences if e["id"] == ev_id), None)
            if ev_data is None:
                continue

            result = ev_data.get("result", {})
            spec = self._build_spec(chart_type, chart_def, result, title)
            if spec:
                spec["evidence_id"] = ev_id
                charts.append(spec)

        ev = self.store.record_evidence(
            run_id=run_id,
            kind="viz",
            produced_by="viz_agent",
            dataset_hash=dataset_hash,
            code="chart_collection",
            params={},
            result={"charts": charts},
            columns=[],
            status="ok",
            depends_on=depends_on,
        )
        return ev.id

    def _build_spec(self, chart_type: str, chart_def: Dict, result: Dict, title: str) -> Dict:
        try:
            if chart_type == "bar":
                data = result.get("data", [])
                x = chart_def.get("x", "")
                y = chart_def.get("y", "")
                if data and x and y:
                    return viz_tools.bar_chart(data, x=x, y=y, title=title)

            elif chart_type == "line":
                data = result.get("data", [])
                x = chart_def.get("x", "")
                y = chart_def.get("y", "")
                if data and x and y:
                    return viz_tools.line_chart(data, x=x, y=y, title=title)

            elif chart_type == "scatter":
                data = result.get("data", [])
                x = chart_def.get("x", "")
                y = chart_def.get("y", "")
                if data and x and y:
                    return viz_tools.scatter_chart(data, x=x, y=y, title=title)

            elif chart_type == "heatmap":
                matrix = result.get("matrix", {})
                if matrix:
                    return viz_tools.heatmap(matrix, title=title)

            elif chart_type == "waterfall":
                contribs = {
                    "Customer effect": result.get("contribution_customers", 0),
                    "ARPU effect": result.get("contribution_arpu", 0),
                    "Interaction": result.get("contribution_interaction", 0),
                }
                return viz_tools.waterfall_chart(contribs, title=title)

        except Exception:
            pass
        return {}
