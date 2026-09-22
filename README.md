# DataPilot

DataPilot is an autonomous, multi-agent AI data scientist.
DataPilot Implementation Plan
DataPilot is an autonomous, multi-agent AI data scientist designed to perform rigorous, reproducible, and statistically justified data analysis.

User Review Required
IMPORTANT

Please review this implementation plan. It covers the full 10-phase build process as described in the prompt. Upon your approval, we will proceed with Phase 1.

Open Questions
WARNING

Do you have a preferred UI color scheme or layout specifically for the Streamlit app to ensure a "premium" feel as per the web application design guidelines, or should we stick to a clean, modern default?
For the Python Sandbox (Phase 3), the prompt mentions using a separate subprocess or Docker container if sandbox.mode: docker. For local development during this session, is it acceptable to default to subprocess mode to simplify environment dependencies while ensuring the AST allowlist and OS-level restrictions are tightly enforced?
Proposed Architecture and Phases
The project will be built in 10 sequential phases. We will not proceed to the next phase until the previous one is fully tested and accepted.

1. Foundation
Setup repository structure (pyproject.toml, Makefile, etc.).
Implement core Pydantic models in ledger/models.py.
Implement append-only SQLite ledger with hash chaining (ledger/store.py).
Implement the safe placeholder resolver (ledger/resolver.py).
Implement provenance DAG (ledger/provenance.py).
Implement replay mechanism (ledger/replay.py).
2. Data Layer
Implement loaders for CSV, Excel, Parquet, and Postgres (ingestion/loaders.py).
Implement snapshot logic using DuckDB (ingestion/snapshot.py).
Implement schema profiling (ingestion/profiler.py).
Implement column role inference (ingestion/roles.py).
3. Sandbox
Implement SQL guard using sqlglot (sandbox/sql_guard.py) and executor (sandbox/sql_executor.py).
Implement Python AST guard (sandbox/python_guard.py) and subprocess runner (sandbox/python_runner.py).
4. Deterministic Tools
Implement data validation (tools/validation.py).
Implement EDA tools (tools/eda.py) and decomposition (tools/decomposition.py).
Implement the statistics allowlist registry (tools/stats/).
Implement ML tools (tools/ml.py).
Implement visualization logic (tools/viz.py).
5. LLM Layer + Agents
Implement Anthropic and mock clients (llm/).
Develop prompts for all agents (prompts/).
Implement the Planner, Data Analyst, SQL, Stats, ML, and Viz agents (agents/).
6. Evaluator, Report & Firewall
Implement Evaluator using rules from thresholds.yaml (agents/evaluator.py).
Implement Report Agent (agents/report_agent.py) and rendering (report/).
Implement Hallucination Firewall (firewall/firewall.py) with all 12 checks.
7. Orchestration
Implement LangGraph workflow (graph/workflow.py).
Implement multi-turn follow-ups (session/followup.py).
8. Benchmark
Create synthetic data generator (benchmark/synth.py).
Implement naive baseline and runner to compute metrics (benchmark/).
9. UI & Export
Build Streamlit application (ui/app.py).
Implement PDF export and reproducibility bundle.
Browser agent verification.
10. Polish
Complete Typer CLI (cli.py).
Write documentation (ARCHITECTURE.md, THREAT_MODEL.md, DECISIONS.md).
Finalize Dockerization and CI pipeline.
Verification Plan
Automated Tests
pytest for all unit and integration tests (target >= 85% coverage).
Adversarial tests specifically targeting the Hallucination Firewall and Sandboxes.
Benchmark suite comparing DataPilot against a naive baseline.
Manual Verification
We will use the browser agent in Phase 9 to verify the UI.
Running make demo to ensure the end-to-end pricing change scenario runs flawlessly and rejects hallucinated claims.

