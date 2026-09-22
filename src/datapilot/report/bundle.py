"""
Reproducibility Bundle Exporter — packages report, ledger DB, environment metadata, and replay script into a ZIP archive.
"""
import io
import json
import zipfile
import platform
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from datapilot.ledger.store import LedgerStore


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
    3. environment.json (Python version, OS, core dependency versions)
    4. replay_instructions.txt (CLI command to verify cryptographic integrity and replay)
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
            # Query from store if not in report_data
            cursor = store.conn.cursor()
            cursor.execute("SELECT * FROM evidence WHERE run_id = ? ORDER BY id ASC", (run_id,))
            rows = [dict(r) for r in cursor.fetchall()]
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

        # 3. Environment metadata
        env_meta = {
            "run_id": run_id,
            "dataset_hash": dataset_hash,
            "python_version": sys.version,
            "platform": platform.platform(),
            "chain_valid": store.verify_chain(run_id),
            "evidence_count": len(evidences),
        }
        zf.writestr("environment.json", json.dumps(env_meta, indent=2).encode("utf-8"))

        # 4. Instructions
        instructions = f"""DataPilot Reproducibility Bundle
Run ID: {run_id}
Dataset SHA-256: {dataset_hash}

To audit and replay this analysis:
1. Extract this zip file.
2. Inspect 'ledger_export.json' for the exact SQL, Python, and statistical tool calls executed.
3. Every evidence record is linked by a cryptographic SHA-256 hash chain ('prev_hash' -> 'hash').
4. To verify integrity, run:
   python -m datapilot.ledger.replay --ledger ledger_export.json --run-id {run_id}
"""
        zf.writestr("README.txt", instructions.encode("utf-8"))

        # Extra files (e.g. raw parquet if provided)
        if extra_files:
            for fname, fbytes in extra_files.items():
                zf.writestr(fname, fbytes)

    buffer.seek(0)
    return buffer.getvalue()
