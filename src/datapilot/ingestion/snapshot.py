import duckdb
import hashlib
from pathlib import Path
from typing import List, Dict

class SnapshotManager:
    def __init__(self, base_dir: str | Path = ".datapilot/snapshots"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _compute_table_hash(self, conn: duckdb.DuckDBPyConnection, table_name: str) -> str:
        # Get schema hash
        schema_query = f"PRAGMA table_info('{table_name}')"
        schema_info = conn.execute(schema_query).fetchall()
        schema_str = "|".join([f"{row[1]}:{row[2]}" for row in schema_info])
        
        # Get data hash: sort-independent canonical hashing
        # 1. hash each row
        # 2. sort row hashes
        # 3. aggregate
        columns = [row[1] for row in schema_info]
        # We cast everything to VARCHAR and concat with a separator to hash a row
        concat_expr = " || '|' || ".join([f"COALESCE(CAST(\"{c}\" AS VARCHAR), '')" for c in columns])
        
        # We use md5 for row hashes, then string_agg ordered by row hash
        query = f"""
            WITH row_hashes AS (
                SELECT md5({concat_expr}) as rh
                FROM "{table_name}"
            )
            SELECT md5(string_agg(rh, '' ORDER BY rh)) 
            FROM row_hashes
        """
        data_hash = conn.execute(query).fetchone()[0] or ""
        
        final_hash = hashlib.sha256(f"{schema_str}||{data_hash}".encode('utf-8')).hexdigest()
        return final_hash

    def create_snapshot(self, conn: duckdb.DuckDBPyConnection, tables: List[str]) -> str:
        table_hashes = []
        for table in sorted(tables):
            table_hashes.append(f"{table}:{self._compute_table_hash(conn, table)}")
            
        combined = "|".join(table_hashes)
        snapshot_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
        
        snapshot_dir = self.base_dir / snapshot_hash
        
        if not snapshot_dir.exists():
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            for table in tables:
                parquet_path = snapshot_dir / f"{table}.parquet"
                # Export table to parquet
                conn.execute(f"COPY \"{table}\" TO '{parquet_path}' (FORMAT PARQUET)")
                
        return snapshot_hash

    def load_snapshot(self, snapshot_hash: str) -> duckdb.DuckDBPyConnection:
        snapshot_dir = self.base_dir / snapshot_hash
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"Snapshot {snapshot_hash} not found at {snapshot_dir}")
            
        conn = duckdb.connect(":memory:")
        for parquet_file in snapshot_dir.glob("*.parquet"):
            table_name = parquet_file.stem
            conn.execute(f"CREATE TABLE \"{table_name}\" AS SELECT * FROM '{parquet_file}'")
            
        return conn
