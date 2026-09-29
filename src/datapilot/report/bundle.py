"""
Reproducibility Bundle Exporter — packages report, ledger DB, environment metadata,
runnable code, and replay script into a ZIP archive.
"""
import io
import json
import zipfile
import platform
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from datapilot.ledger.store import LedgerStore


def _build_runnable_code(evidences: list, run_id: str, dataset_hash: str) -> str:
    """Assemble a single runnable Python script from all exportable evidence code blocks."""
    header = f'''#!/usr/bin/env python3
"""
DataPilot Reproducibility Script
Run ID: {run_id}
Dataset SHA-256: {dataset_hash}

Auto-generated from the DataPilot ledger. Every code block below corresponds to
one evidence record that was executed and cryptographically logged.

Usage:
    pip install pandas duckdb scipy pingouin statsmodels matplotlib seaborn
    python runnable_code.py
"""
import pandas as pd
import numpy as np

# ─── Load your dataset ─────────────────────────────────────────────────────────
# Replace this with the actual path to your dataset parquet/csv file.
dataset_path = "dataset.parquet"  # or "dataset.csv"
try:
    df = pd.read_parquet(dataset_path)
except Exception:
    try:
        df = pd.read_csv(dataset_path.replace(".parquet", ".csv"))
    except Exception:
        print("⚠️  Could not load dataset. Set `dataset_path` to your data file.")
        df = pd.DataFrame()

print(f"Dataset loaded: {{len(df)}} rows × {{len(df.columns)}} columns")
print()
'''
    blocks = []
    for ev in evidences:
        code = ev.get("code", "")
        if not code or code in ("synthesis", "report_synthesis", "evaluator_threshold_checks"):
            continue
        eid = ev.get("id", "?")
        purpose = ev.get("purpose") or ev.get("kind", "")
        order = ev.get("execution_order", "?")
        block = f"""
# ══════════════════════════════════════════════════════════════════════════════
# Evidence {eid}  |  Step {order}  |  {purpose}
# ══════════════════════════════════════════════════════════════════════════════
{code}
"""
        blocks.append(block)

    if not blocks:
        return header + "\n# No exportable code blocks found in this run.\n"
    return header + "\n".join(blocks)


def create_reproducibility_bundle(
    run_id: str,
    store: LedgerStore,
    report_data: Dict[str, Any],
    dataset_hash: str,
    extra_files: Optional[Dict[str, bytes]] = None,
) -> bytes:
    """
    Creates an in-memory ZIP archive containing everything required to audit and reproduce a run:
    1. report.md & report.html
    2. ledger_export.json (all evidence records for run)
    3. runnable_code.py (assembled executable script)
    4. environment.json (Python version, OS, core dependency versions)
    5. README.txt (replay instructions)
    """
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Report files
        md_content = report_data.get("markdown", "")
        html_content = report_data.get("html", "")
        zf.writestr("report.md", md_content.encode("utf-8"))
        zf.writestr("report.html", html_content.encode("utf-8"))

        # 2. Evidence dump from Ledger
        evidences = report_data.get("evidences", [])
        if not evidences:
            rows = store.list_evidence(run_id)
            for r in rows:
                for k in ["depends_on", "params", "result", "columns", "lib_versions"]:
                    if isinstance(r.get(k), str):
                        try:
                            r[k] = json.loads(r[k])
                        except Exception:
                            pass
            evidences = rows

        ledger_json = json.dumps(evidences, indent=2, default=str)
        zf.writestr("ledger_export.json", ledger_json.encode("utf-8"))

        # 3. Runnable code script
        runnable = _build_runnable_code(evidences, run_id, dataset_hash)
        zf.writestr("runnable_code.py", runnable.encode("utf-8"))

        # 4. Environment metadata
        env_meta = {
            "run_id": run_id,
            "dataset_hash": dataset_hash,
            "python_version": sys.version,
            "platform": platform.platform(),
            "chain_valid": store.verify_chain(run_id),
            "evidence_count": len(evidences),
        }
        zf.writestr("environment.json", json.dumps(env_meta, indent=2).encode("utf-8"))

        # 5. Instructions
        instructions = f"""DataPilot Reproducibility Bundle
=================================
Run ID: {run_id}
Dataset SHA-256: {dataset_hash}

Contents:
  report.md           — Human-readable Markdown report
  report.html         — Standalone HTML report (open in browser)
  ledger_export.json  — All evidence records (cryptographically hash-chained)
  runnable_code.py    — Executable Python script reproducing all analyses
  environment.json    — Environment metadata & chain validity

To reproduce this analysis:
1. Install dependencies:
   pip install pandas duckdb scipy pingouin statsmodels matplotlib seaborn
2. Place your dataset at the path referenced in runnable_code.py
3. Run: python runnable_code.py

To audit cryptographic integrity:
   python -m datapilot.ledger.replay --ledger ledger_export.json --run-id {run_id}

Hash chain verification result: {env_meta['chain_valid']}
"""
        zf.writestr("README.txt", instructions.encode("utf-8"))

        # Extra files (e.g. raw parquet if provided)
        if extra_files:
            for fname, fbytes in extra_files.items():
                zf.writestr(fname, fbytes)

    buffer.seek(0)
    return buffer.getvalue()
