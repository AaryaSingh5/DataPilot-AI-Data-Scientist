import duckdb
from pathlib import Path
from typing import List, Optional

def load_file(file_path: str | Path, table_name: str = "dataset") -> duckdb.DuckDBPyConnection:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    conn = duckdb.connect(":memory:")
    ext = path.suffix.lower()
    
    if ext == ".csv":
        conn.execute(f"CREATE TABLE \"{table_name}\" AS SELECT * FROM read_csv_auto('{path}', all_varchar=false)")
    elif ext == ".tsv":
        conn.execute(f"CREATE TABLE \"{table_name}\" AS SELECT * FROM read_csv_auto('{path}', delim='\t', all_varchar=false)")
    elif ext == ".parquet":
        conn.execute(f"CREATE TABLE \"{table_name}\" AS SELECT * FROM read_parquet('{path}')")
    elif ext in [".xlsx", ".xls"]:
        conn.execute("INSTALL spatial")
        conn.execute("LOAD spatial")
        conn.execute(f"CREATE TABLE \"{table_name}\" AS SELECT * FROM st_read('{path}')")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
        
    return conn

def load_postgres(uri: str, tables: List[str]) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL postgres")
    conn.execute("LOAD postgres")
    conn.execute(f"ATTACH '{uri}' AS pg (TYPE postgres, READ_ONLY)")
    
    for table in tables:
        # Copy to memory
        conn.execute(f"CREATE TABLE \"{table}\" AS SELECT * FROM pg.\"{table}\"")
        
    conn.execute("DETACH pg")
    return conn
