import pytest
import duckdb
from pathlib import Path
from datapilot.ingestion.snapshot import SnapshotManager
from datapilot.ingestion.loaders import load_file
from datapilot.ingestion.profiler import profile_table
from datapilot.ingestion.roles import infer_roles

def test_snapshot_determinism(tmp_path: Path):
    manager = SnapshotManager(base_dir=tmp_path / "snapshots")
    
    # Create an in-memory DB and add some data
    conn = duckdb.connect()
    conn.execute("CREATE TABLE test_table (id INT, val VARCHAR)")
    conn.execute("INSERT INTO test_table VALUES (1, 'a'), (2, 'b')")
    
    hash1 = manager.create_snapshot(conn, ["test_table"])
    
    # Recreate the exact same data but inserted in different order
    conn2 = duckdb.connect()
    conn2.execute("CREATE TABLE test_table (id INT, val VARCHAR)")
    conn2.execute("INSERT INTO test_table VALUES (2, 'b'), (1, 'a')")
    
    hash2 = manager.create_snapshot(conn2, ["test_table"])
    
    # The snapshot hash must be deterministic (sort-independent)
    assert hash1 == hash2
    
    # Modify a cell
    conn3 = duckdb.connect()
    conn3.execute("CREATE TABLE test_table (id INT, val VARCHAR)")
    conn3.execute("INSERT INTO test_table VALUES (1, 'c'), (2, 'b')")
    
    hash3 = manager.create_snapshot(conn3, ["test_table"])
    
    assert hash1 != hash3

def test_load_csv_and_profile(tmp_path: Path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("date,revenue,customer_id,region\n2023-01-01,100,c1,NA\n2023-01-02,150,c2,EU\n2023-01-03,,c3,NA\n")
    
    conn = load_file(csv_file, "dataset")
    
    profile = profile_table(conn, "dataset")
    
    assert "revenue" in profile
    assert profile["revenue"]["null_rate"] == 1/3
    assert profile["date"]["semantic_type"] == "date"
    assert profile["region"]["semantic_type"] == "category"
    
    roles = infer_roles(profile)
    assert roles["date"] == "date"
    assert roles["revenue"] == "metric"
    assert roles["customer_id"] == "customer_id"
    assert roles["region"] == "region"
