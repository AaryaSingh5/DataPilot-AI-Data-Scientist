"""
Tests for Phase 9: Reproducibility Bundle Exporter.
"""
import zipfile
import io
import json
import pytest

from datapilot.ledger.store import LedgerStore
from datapilot.report.bundle import create_reproducibility_bundle


@pytest.fixture
def store_with_run(tmp_path):
    store = LedgerStore(tmp_path / "ledger.db")
    store.record_evidence(
        run_id="run_test_bundle",
        kind="plan",
        produced_by="planner",
        dataset_hash="hash_bundle_123",
        code="plan",
        params={"q": "test"},
        result={"hypotheses": []},
        columns=[],
        status="ok",
    )
    return store


def test_reproducibility_bundle_contents(store_with_run):
    report_data = {
        "run_id": "run_test_bundle",
        "markdown": "# Test Report\nFindings here.",
        "html": "<!DOCTYPE html><html><body>Report</body></html>",
        "evidences": [
            {
                "id": "ev_0001",
                "kind": "plan",
                "status": "ok",
                "result": {},
            }
        ],
    }

    bundle_bytes = create_reproducibility_bundle(
        run_id="run_test_bundle",
        store=store_with_run,
        report_data=report_data,
        dataset_hash="hash_bundle_123",
    )

    assert isinstance(bundle_bytes, bytes)
    assert len(bundle_bytes) > 0

    # Unpack ZIP in memory and inspect files
    zf = zipfile.ZipFile(io.BytesIO(bundle_bytes))
    file_list = zf.namelist()

    assert "report.md" in file_list
    assert "report.html" in file_list
    assert "ledger_export.json" in file_list
    assert "environment.json" in file_list
    assert "README.txt" in file_list

    # Verify environment metadata
    env_content = json.loads(zf.read("environment.json").decode("utf-8"))
    assert env_content["run_id"] == "run_test_bundle"
    assert env_content["chain_valid"] is True
