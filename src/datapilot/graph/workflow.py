"""
LangGraph Workflow — coordinates Planner, Specialists, Analyst, Evaluator, Viz, and Report agents.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import duckdb
from langgraph.graph import StateGraph, START, END

from datapilot.graph.state import DataPilotState
from datapilot.llm.client import LLMClient, LLMBudgetExceeded
from datapilot.ledger.store import LedgerStore
from datapilot.agents.planner import PlannerAgent
from datapilot.agents.sql_agent import SQLAgent
from datapilot.agents.stats_agent import StatsAgent
from datapilot.agents.ml_agent import MLAgent
from datapilot.agents.viz_agent import VizAgent
from datapilot.agents.analyst import AnalystAgent
from datapilot.agents.evaluator import EvaluatorAgent
from datapilot.agents.report_agent import ReportAgent


MAX_EXECUTOR_STEPS = 5


class DataPilotWorkflow:
    def __init__(self, llm: LLMClient, store: LedgerStore, snapshot_dir: Path):
        self.llm = llm
        self.store = store
        self.snapshot_dir = Path(snapshot_dir)

        # Initialize agents
        self.planner = PlannerAgent(llm=llm, store=store)
        self.sql_agent = SQLAgent(llm=llm, store=store, snapshot_dir=self.snapshot_dir)
        self.stats_agent = StatsAgent(llm=llm, store=store)
        self.ml_agent = MLAgent(llm=llm, store=store, snapshot_dir=self.snapshot_dir)
        self.viz_agent = VizAgent(llm=llm, store=store)
        self.analyst = AnalystAgent(llm=llm, store=store)
        self.evaluator = EvaluatorAgent(store=store)
        self.report_agent = ReportAgent(store=store)

        self.graph = self._build_graph()

    def _setup_duckdb(self, dataset_hash: str) -> tuple[duckdb.DuckDBPyConnection, str]:
        conn = duckdb.connect(":memory:")
        snap_dir = self.snapshot_dir / dataset_hash
        table_name = "data"
        if snap_dir.exists():
            pq_files = list(snap_dir.glob("*.parquet"))
            if pq_files:
                table_name = pq_files[0].stem
                for pq in pq_files:
                    conn.execute(f'CREATE VIEW "{pq.stem}" AS SELECT * FROM read_parquet(\'{pq}\')')
        return conn, table_name

    def _build_graph(self):
        builder = StateGraph(DataPilotState)

        # Node: Planner
        def planner_node(state: DataPilotState) -> Dict[str, Any]:
            print(" -> Running Planner Node...")
            plan = self.planner.run(
                run_id=state["run_id"],
                dataset_hash=state["dataset_hash"],
                question=state["question"],
                schema=state["schema"],
                role_map=state["role_map"],
            )
            eids = list(state.get("evidence_ids", []))
            if "_evidence_id" in plan:
                eids.append(plan["_evidence_id"])
            print(f" -> Planner completed. Hypotheses: {len(plan.get('hypotheses', []))}, Steps: {len(plan.get('plan', []))}")
            return {"plan": plan, "evidence_ids": eids}

        # Node: Executor
        def executor_node(state: DataPilotState) -> Dict[str, Any]:
            print(" -> Running Executor Node...")
            plan = state.get("plan", {})
            steps = plan.get("plan", [])[:MAX_EXECUTOR_STEPS]  # Max steps guard
            hypotheses = plan.get("hypotheses", [])
            hyp_map = {h.get("id"): h for h in hypotheses if h.get("id")}
            eids = list(state.get("evidence_ids", []))

            conn, default_table = self._setup_duckdb(state["dataset_hash"])
            table_name = state.get("table_name") or default_table

            for step in steps:
                agent_type = step.get("agent", "").lower()
                hid = step.get("hypothesis_id", "")
                hyp = hyp_map.get(
                    hid,
                    {
                        "id": hid,
                        "statement": step.get("action", ""),
                        "columns": list(state["role_map"].keys()),
                        "test_type": "before_after",
                    },
                )
                action_desc = step.get("action", "")

                try:
                    if agent_type == "sql":
                        ev_id = self.sql_agent.run(
                            run_id=state["run_id"],
                            dataset_hash=state["dataset_hash"],
                            question=action_desc,
                            schema=state["schema"],
                            role_map=state["role_map"],
                            hypothesis_id=hid,
                            depends_on=eids[-1:] if eids else None,
                        )
                        eids.append(ev_id)

                    elif agent_type == "stats":
                        ev_id = self.stats_agent.run(
                            run_id=state["run_id"],
                            dataset_hash=state["dataset_hash"],
                            hypothesis=hyp,
                            conn=conn,
                            table_name=table_name,
                            depends_on=eids[-1:] if eids else None,
                        )
                        eids.append(ev_id)

                    elif agent_type == "ml":
                        ev_id = self.ml_agent.run(
                            run_id=state["run_id"],
                            dataset_hash=state["dataset_hash"],
                            hypothesis=hyp,
                            conn=conn,
                            table_name=table_name,
                            depends_on=eids[-1:] if eids else None,
                        )
                        eids.append(ev_id)
                except LLMBudgetExceeded:
                    raise
                except Exception:
                    pass

            # Gather all evidence dicts
            ev_dicts = []
            for eid in eids:
                ev_obj = self.store.get_evidence(eid)
                if ev_obj:
                    ev_dicts.append(ev_obj.model_dump())

            return {"evidence_ids": eids, "evidences": ev_dicts, "table_name": table_name}

        # Node: Analyst
        def analyst_node(state: DataPilotState) -> Dict[str, Any]:
            print(" -> Running Analyst Node...")
            evidences = state.get("evidences", [])
            eids = list(state.get("evidence_ids", []))
            plan = state.get("plan", {})

            analysis_ev_id = self.analyst.run(
                run_id=state["run_id"],
                dataset_hash=state["dataset_hash"],
                plan=plan,
                evidences=evidences,
                depends_on=eids,
            )
            eids.append(analysis_ev_id)
            analysis_ev = self.store.get_evidence(analysis_ev_id)
            analysis = analysis_ev.result if analysis_ev else {}
            if analysis_ev:
                evidences.append(analysis_ev.model_dump())

            print(" -> Analyst Node completed.")
            return {"analysis": analysis, "evidence_ids": eids, "evidences": evidences}

        # Node: Evaluator
        def evaluator_node(state: DataPilotState) -> Dict[str, Any]:
            print(" -> Running Evaluator Node...")
            evidences = state.get("evidences", [])
            eids = list(state.get("evidence_ids", []))

            eval_res = self.evaluator.run(
                run_id=state["run_id"],
                dataset_hash=state["dataset_hash"],
                evidences=evidences,
                depends_on=eids,
            )
            if "_evidence_id" in eval_res:
                eids.append(eval_res["_evidence_id"])
                ev_obj = self.store.get_evidence(eval_res["_evidence_id"])
                if ev_obj:
                    evidences.append(ev_obj.model_dump())

            print(f" -> Evaluator Node completed. Overall score: {eval_res.get('overall_score')}")
            return {"evaluation": eval_res, "evidence_ids": eids, "evidences": evidences}

        # Node: Visualization
        def viz_node(state: DataPilotState) -> Dict[str, Any]:
            print(" -> Running Visualization Node...")
            evidences = state.get("evidences", [])
            eids = list(state.get("evidence_ids", []))

            viz_ev_id = self.viz_agent.run(
                run_id=state["run_id"],
                dataset_hash=state["dataset_hash"],
                evidences=evidences,
                depends_on=eids,
            )
            eids.append(viz_ev_id)
            viz_ev = self.store.get_evidence(viz_ev_id)
            charts = viz_ev.result.get("charts", []) if viz_ev else []
            if viz_ev:
                evidences.append(viz_ev.model_dump())

            print(f" -> Visualization Node completed. Generated {len(charts)} chart(s).")
            return {"charts": charts, "evidence_ids": eids, "evidences": evidences}

        # Node: Report
        def report_node(state: DataPilotState) -> Dict[str, Any]:
            print(" -> Running Report Node...")
            eids = list(state.get("evidence_ids", []))
            report = self.report_agent.run(
                run_id=state["run_id"],
                dataset_hash=state["dataset_hash"],
                question=state["question"],
                plan=state.get("plan", {}),
                analysis=state.get("analysis", {}),
                evaluation=state.get("evaluation", {}),
                charts=state.get("charts", []),
                evidences=state.get("evidences", []),
                schema=state.get("schema"),
                depends_on=eids,
            )
            if "_evidence_id" in report:
                eids.append(report["_evidence_id"])

            print(f" -> Report Node completed. Firewall Passed: {report.get('firewall', {}).get('passed')}")
            return {
                "report": report,
                "firewall_passed": report.get("firewall", {}).get("passed", False),
                "evidence_ids": eids,
            }

        # Wire graph
        builder.add_node("planner", planner_node)
        builder.add_node("executor", executor_node)
        builder.add_node("analyst", analyst_node)
        builder.add_node("evaluator", evaluator_node)
        builder.add_node("viz", viz_node)
        builder.add_node("report", report_node)

        builder.add_edge(START, "planner")
        builder.add_edge("planner", "executor")
        builder.add_edge("executor", "analyst")
        builder.add_edge("analyst", "evaluator")
        builder.add_edge("evaluator", "viz")
        builder.add_edge("viz", "report")
        builder.add_edge("report", END)

        return builder.compile()

    def run(self, initial_state: DataPilotState) -> DataPilotState:
        try:
            return self.graph.invoke(initial_state)
        except LLMBudgetExceeded as exc:
            err_msg = str(exc)
            errors = list(initial_state.get("errors", []))
            errors.append(err_msg)
            # Produce fallback state surfacing budget exceeded
            fallback_report = {
                "title": f"DataPilot Report (Halted: Budget Exceeded)",
                "run_id": initial_state.get("run_id", "run_error"),
                "dataset_hash": initial_state.get("dataset_hash", ""),
                "question": initial_state.get("question", ""),
                "executive_summary": f"Execution halted: {err_msg}",
                "plan": initial_state.get("plan", {}),
                "conclusions": [],
                "evaluation": {"overall_score": 0.0, "status": "halted"},
                "charts": [],
                "evidences": initial_state.get("evidences", []),
                "firewall": {"passed": False, "violations": [], "warnings": []},
                "markdown": f"# DataPilot Report\n\n**Execution Halted**: {err_msg}",
                "html": f"<h1>DataPilot Report</h1><p><strong>Execution Halted</strong>: {err_msg}</p>",
            }
            res_state: DataPilotState = {
                **initial_state,
                "report": fallback_report,
                "errors": errors,
                "firewall_passed": False,
            }
            return res_state
