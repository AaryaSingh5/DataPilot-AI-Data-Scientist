"""
End-to-end runner script for DataPilot workflow using NVIDIA LLM provider on synthetic benchmark.
"""
import os
import sys
import uuid
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datapilot.benchmark.synth import generate_pricing_experiment
from datapilot.ingestion.snapshot import SnapshotManager
from datapilot.ingestion.profiler import profile_table
from datapilot.ingestion.roles import infer_roles
from datapilot.ledger.store import LedgerStore
from datapilot.llm.client import get_llm_client, NvidiaClient
from datapilot.graph.workflow import DataPilotWorkflow
from datapilot.graph.state import DataPilotState
from datapilot.report.bundle import create_reproducibility_bundle
import duckdb


def main():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: NVIDIA_API_KEY environment variable is not set.")
        sys.exit(1)

    print("=== DataPilot End-to-End NVIDIA Benchmark Run ===")
    
    # 1. Prepare temp cache directories
    cache_dir = Path(".datapilot_cache_benchmark")
    cache_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = cache_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    ledger_db = cache_dir / "ledger.db"

    store = LedgerStore(db_path=ledger_db)

    # 2. Generate pricing experiment benchmark dataset
    print("Generating synthetic pricing experiment dataset (1200 rows)...")
    df, gt = generate_pricing_experiment(n_samples=1200, seed=42)

    # 3. Snapshot & Profile dataset in DuckDB
    conn = duckdb.connect(":memory:")
    conn.register("data", df)
    mgr = SnapshotManager(snap_dir)
    d_hash = mgr.create_snapshot(conn, ["data"])
    col_profiles = profile_table(conn, "data")
    roles = infer_roles(col_profiles)

    dtypes = {col: str(df[col].dtype) for col in df.columns}
    schema = {"data": dtypes}

    print(f"Dataset snapshot hash: {d_hash}")

    # 4. Instantiate Nvidia Client
    llm = get_llm_client(provider="nvidia", api_key=api_key)
    print(f"LLM Client: NvidiaClient (Model: {getattr(llm, 'model', 'default')})")

    # 5. Build and execute workflow
    wf = DataPilotWorkflow(llm=llm, store=store, snapshot_dir=snap_dir)

    run_id = f"run_bench_{uuid.uuid4().hex[:8]}"
    question = "Did the pricing change increase revenue, and is the effect confounded by customer segment?"

    initial_state: DataPilotState = {
        "run_id": run_id,
        "dataset_hash": d_hash,
        "question": question,
        "schema": schema,
        "role_map": roles,
        "table_name": "data",
        "snapshot_dir": str(snap_dir),
        "evidence_ids": [],
        "evidences": [],
    }

    print(f"\nRunning DataPilotWorkflow for run_id={run_id}...")
    final_state = wf.run(initial_state)

    report = final_state.get("report", {})
    errors = final_state.get("errors", [])

    print("\n=== Workflow Execution Completed ===")
    if errors:
        print(f"Workflow Errors: {errors}")

    print(f"Report Title: {report.get('title')}")
    print(f"Executive Summary: {report.get('executive_summary')}")
    
    fw_info = report.get("firewall", {})
    print(f"Firewall Passed: {fw_info.get('passed')}")
    if fw_info.get("violations"):
        print("Firewall Violations:")
        for v in fw_info.get("violations"):
            print(f"  - [{v.get('check_name')}] Claim {v.get('claim_id')}: {v.get('message')}")

    # 6. Verify ledger integrity
    is_valid = store.verify_chain(run_id)
    print(f"\nLedger SHA-256 Chain Verified: {is_valid}")

    # 7. Create Reproducibility Bundle
    bundle_zip = create_reproducibility_bundle(
        run_id=run_id,
        store=store,
        report_data=report,
        dataset_hash=d_hash,
    )
    bundle_path = cache_dir / f"bundle_{run_id}.zip"
    bundle_path.write_bytes(bundle_zip)
    print(f"Reproducibility bundle saved to: {bundle_path} ({len(bundle_zip)} bytes)")

    if report.get("title") and is_valid and len(bundle_zip) > 0 and not errors and fw_info.get("passed"):
        print("\nSUCCESS: End-to-end run completed with valid ledger, passed firewall, and generated bundle!")
        sys.exit(0)
    else:
        print("\nFAILURE or WARNING: Issues encountered during run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
