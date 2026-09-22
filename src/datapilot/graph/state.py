"""
LangGraph State definition for DataPilot workflow.
"""
from typing import TypedDict, List, Dict, Any, Optional


class DataPilotState(TypedDict, total=False):
    run_id: str
    dataset_hash: str
    question: str
    schema: Dict[str, Any]
    role_map: Dict[str, str]
    table_name: str
    snapshot_dir: str

    plan: Dict[str, Any]
    evidence_ids: List[str]
    evidences: List[Dict[str, Any]]

    analysis: Dict[str, Any]
    evaluation: Dict[str, Any]
    charts: List[Dict[str, Any]]
    report: Dict[str, Any]
    firewall_passed: bool
    errors: List[str]
