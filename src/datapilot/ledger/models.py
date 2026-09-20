from typing import Any, Literal
from pydantic import BaseModel, ConfigDict
from datetime import datetime

class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str                          # "ev_0001", sequential per run
    run_id: str
    kind: Literal["plan", "validation", "sql", "stat_test", "model", "plot"]
    produced_by: Literal["planner", "analyst", "sql", "stats", "ml", "viz", "system"]
    depends_on: list[str] = []       # evidence IDs this was derived from (provenance DAG)
    dataset_hash: str                # sha256 of the snapshot it ran on
    code: str                        # exact SQL / Python / tool-call spec executed
    params: dict[str, Any]           # seed, filters, controls, alpha, test name, etc.
    result: dict[str, Any]           # machine-readable numbers, JSON-serializable
    columns: list[str]               # dataset columns touched
    artifact_path: str | None        # parquet result table or plot file
    artifact_hash: str | None
    lib_versions: dict[str, str]
    status: Literal["ok", "error"]
    error: str | None
    created_at: datetime
    prev_hash: str                   # hash chain for tamper evidence
    hash: str                        # sha256 over all fields + prev_hash

class Hypothesis(BaseModel):
    id: str                          # "h1"
    statement: str                   # human-readable, contains no numbers
    columns: list[str]
    test_type: Literal["before_after", "association", "controlled_association",
                       "segment_contribution", "seasonality", "data_artifact"]
    controls: list[str] = []
    expected_direction: Literal["positive", "negative", "any"]

class PeriodSpec(BaseModel):
    type: str # "quarter", "month", etc.
    n: int
    year: str # "latest", "previous", or specific year string

class Plan(BaseModel):
    question: str
    target_metric: str               # e.g. revenue
    role_map: dict[str, str]         # resolved column roles
    period_spec: PeriodSpec          # structured, NOT resolved by the LLM
    hypotheses: list[Hypothesis]     # at least 4, always including seasonality
                                     # and data_artifact hypotheses
    decomposition_dims: list[str]
    needs_ml: bool

class Claim(BaseModel):
    id: str
    text_template: str               # "Revenue fell {ev_0004.result.pct_change:.1f}%."
    evidence_ids: list[str]
    claim_type: Literal["descriptive", "correlational", "causal"]
    strength: Literal["supported", "weak", "insufficient", "refuted"]
    hypothesis_id: str | None

class Verdict(BaseModel):
    hypothesis_id: str
    verdict: Literal["supported", "weak", "insufficient", "refuted"]
    rule_id: str                     # which rule in thresholds.yaml fired
    evidence_ids: list[str]
    rationale_codes: list[str]       # machine-readable, e.g. ["attenuation_gt_50pct"]
