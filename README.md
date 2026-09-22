# 🧭 DataPilot — Autonomous AI Data Scientist

> **Rigorous, reproducible, and statistically justified data analysis — fully autonomous, multi-agent, and cryptographically auditable.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-51%20passing-brightgreen?logo=pytest)](tests/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?logo=streamlit)](src/datapilot/ui/app.py)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What is DataPilot?

DataPilot is an autonomous, multi-agent AI data scientist that takes a natural-language research question and a dataset, then produces a statistically rigorous, hallucination-firewalled, cryptographically auditable analysis report — with zero manual intervention.

Every step — from SQL queries to statistical tests to LLM conclusions — is recorded in an append-only, SHA-256 hash-chained ledger for full reproducibility and audit.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **Multi-Agent Pipeline** | Planner → SQL → Stats → ML → Viz → Analyst → Evaluator → Report |
| 🛡️ **12-Check Hallucination Firewall** | Blocks fabricated statistics, fake column names, unjustified causal claims, leakage, and more |
| 🔐 **Cryptographic Audit Ledger** | SHA-256 hash-chained SQLite ledger — every evidence record is tamper-evident |
| 🧪 **Deterministic Sandbox** | SQL guard (sqlglot) + Python AST guard prevent prompt injection and code execution attacks |
| 📊 **Statistical Rigour** | Before/after tests, correlation, BH-corrected FDR, power analysis, SHAP explainability |
| 🏋️ **Benchmark Suite** | Synthetic pricing experiment dataset with ground truth; naive baseline comparison |
| 📦 **Reproducibility Bundles** | One-click ZIP export: report + ledger dump + environment metadata + replay instructions |
| 🖥️ **Streamlit UI** | 4-tab dark-mode web app with live execution, report preview, and ledger viewer |

---

## 🏗️ Architecture

```
datapilot/
├── src/datapilot/
│   ├── agents/          # Planner, Analyst, SQL, Stats, ML, Viz agents
│   ├── benchmark/       # Synthetic data generator, naive baseline, runner
│   ├── config/          # thresholds.yaml, settings
│   ├── firewall/        # 12-check Hallucination Firewall
│   ├── graph/           # LangGraph StateGraph workflow & session manager
│   ├── ingestion/       # CSV/Parquet/Excel loaders, DuckDB snapshots, profiler, roles
│   ├── ledger/          # Append-only SQLite store, hash chaining, provenance DAG, replay
│   ├── llm/             # Mock & Anthropic LLM clients
│   ├── prompts/         # Jinja2 prompt templates for all agents
│   ├── report/          # Markdown/HTML renderer + reproducibility bundle exporter
│   ├── sandbox/         # SQL guard, SQL executor, Python AST guard, Python runner
│   ├── session/         # Multi-turn session & follow-up manager
│   ├── tools/           # Validation, EDA, decomposition, stats, ML (SHAP), viz
│   └── ui/              # Streamlit web application (app.py)
└── tests/
    ├── adversarial/     # Firewall + sandbox adversarial tests
    └── unit/            # Agent, tool, ledger, benchmark, workflow unit tests
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
# Using the bundled virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -e ".[dev]"
```

### 2. Run the Streamlit UI

```bash
streamlit run src/datapilot/ui/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run the test suite

```bash
pytest -v
# 51 tests — unit, adversarial, benchmark, workflow
```

---

## 🖥️ Web Application

The Streamlit app provides 4 tabs:

| Tab | Description |
|---|---|
| 🔍 **Investigation & Execution** | Enter a research question and run the full multi-agent workflow |
| 📊 **Dataset Profile & Roles** | Preview data, inferred column roles, and summary statistics |
| 📑 **Final Report & Firewall** | Download Markdown/HTML reports and reproducibility ZIP bundle |
| 🛡️ **Audit Ledger & Provenance** | Browse the cryptographic hash-chained evidence ledger |

**Sidebar options:**
- Switch between the **synthetic benchmark** (SaaS pricing experiment) or **upload your own** CSV/Parquet/Excel
- Configure **significance alpha (α)** and **FDR q-value**
- Select **Mock (deterministic)** or **Anthropic Claude** LLM backend

---

## 🛡️ Hallucination Firewall — 12 Checks

The firewall runs after every agent conclusion before it enters the final report:

| # | Check | Blocks |
|---|---|---|
| 1 | **Provenance** | Claims referencing non-existent evidence IDs |
| 2 | **Numerical Exactness** | Numbers that differ from ledger values by > tolerance |
| 3 | **Schema Integrity** | References to columns not in the dataset |
| 4 | **Causal Language** | "causes", "leads to" without a controlled experiment |
| 5 | **Significance** | Claims of significance where p > α |
| 6 | **FDR** | Multiple-comparison claims that fail BH correction |
| 7 | **Effect Size** | Exaggerated effect size claims vs. Cohen's d |
| 8 | **Directionality** | Sign inversions (positive claimed, negative measured) |
| 9 | **Sample Size** | Fabricated N that doesn't match the actual row count |
| 10 | **Leakage** | Target variable used as a model feature |
| 11 | **Scope** | Universal claims made on filtered/subset data |
| 12 | **Non-Vacuous** | Tautological or empty claims ("the data shows the data") |

---

## 📦 Reproducibility Bundle

Every completed run can be exported as a ZIP containing:

- `report.md` — Full Markdown analysis report
- `report.html` — Standalone HTML report
- `ledger_export.json` — All evidence records with hash chain
- `environment.json` — Python version, OS, chain validity, evidence count
- `README.txt` — Audit and replay instructions

---

## 🧪 Benchmark

DataPilot ships with a synthetic **SaaS pricing experiment** dataset (1,200 rows) with known ground truth:

- **Synthetic generator** (`benchmark/synth.py`) — configurable treatment effect, confounders, and noise
- **Naive baseline** — a simple pre/post mean comparison (deliberately confounded)
- **Benchmark runner** — compares DataPilot's hallucination-firewalled output against the naive baseline and ground truth, computing direction accuracy, p-value accuracy, and effect size error

---

## 📋 Phase Completion

| Phase | Description | Status |
|---|---|---|
| 1 | Foundation — Ledger, resolver, provenance, replay | ✅ Complete |
| 2 | Data Layer — Loaders, snapshots, profiler, roles | ✅ Complete |
| 3 | Sandbox — SQL guard, Python AST guard | ✅ Complete |
| 4 | Deterministic Tools — Validation, EDA, stats, ML, viz | ✅ Complete |
| 5 | LLM Layer + Agents — Planner, Analyst, SQL, Stats, ML, Viz | ✅ Complete |
| 6 | Evaluator, Report & Hallucination Firewall (12 checks) | ✅ Complete |
| 7 | Orchestration — LangGraph workflow, session manager | ✅ Complete |
| 8 | Benchmark — Synthetic data, naive baseline, runner | ✅ Complete |
| 9 | UI & Export — Streamlit app, reproducibility bundle | ✅ Complete |

---

## 🔧 Development

```bash
# Run all tests
make test

# Run with coverage
make coverage

# Lint
make lint
```

---

## 📄 License

MIT © 2026 DataPilot Contributors
