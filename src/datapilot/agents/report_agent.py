"""
Report Agent — compiles validated findings, evaluation scores, and visualizations into a report.
"""
from typing import Dict, Any, List, Optional
from datapilot.ledger.store import LedgerStore
from datapilot.firewall.firewall import HallucinationFirewall, FirewallResult
from datapilot.report.renderer import render_markdown, render_html


class ReportAgent:
    def __init__(self, store: LedgerStore, firewall: Optional[HallucinationFirewall] = None):
        self.store = store
        self.firewall = firewall or HallucinationFirewall()

    def run(
        self,
        run_id: str,
        dataset_hash: str,
        question: str,
        plan: Dict[str, Any],
        analysis: Dict[str, Any],
        evaluation: Dict[str, Any],
        charts: List[Dict[str, Any]],
        evidences: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        conclusions = analysis.get("conclusions", [])
        overall_summary = analysis.get("overall_summary", "")

        # Pass conclusions through Hallucination Firewall
        if schema and not self.firewall.schema:
            self.firewall.schema = schema
        firewall_result: FirewallResult = self.firewall.verify_conclusions(conclusions, evidences)

        report_data = {
            "title": f"DataPilot Investigation Report: {question[:60]}",
            "run_id": run_id,
            "dataset_hash": dataset_hash,
            "question": question,
            "executive_summary": overall_summary,
            "plan": plan,
            "conclusions": conclusions,
            "evaluation": evaluation,
            "charts": charts,
            "evidences": evidences,
            "firewall": {
                "passed": firewall_result.passed,
                "violations": [v.model_dump() for v in firewall_result.violations],
                "warnings": [w.model_dump() for w in firewall_result.warnings],
            },
        }

        markdown_content = render_markdown(report_data)
        html_content = render_html(report_data)

        report_data["markdown"] = markdown_content
        report_data["html"] = html_content

        # Record to ledger
        ev = self.store.record_evidence(
            run_id=run_id,
            kind="report",
            produced_by="report_agent",
            dataset_hash=dataset_hash,
            code="report_synthesis",
            params={"question": question, "firewall_passed": firewall_result.passed},
            result={
                "title": report_data["title"],
                "firewall_passed": firewall_result.passed,
                "violations_count": len(firewall_result.violations),
                "evaluation_score": evaluation.get("overall_score"),
                "markdown_length": len(markdown_content),
            },
            columns=[],
            status="ok" if firewall_result.passed else "error",
            error=None if firewall_result.passed else f"Firewall blocked {len(firewall_result.violations)} claims",
            depends_on=depends_on,
        )
        report_data["_evidence_id"] = ev.id
        return report_data
