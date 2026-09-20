import pytest
from datapilot.sandbox.sql_guard import SQLGuard, SQLGuardError
from datapilot.sandbox.python_guard import PythonGuard, PythonGuardError
import subprocess
import os

def test_sql_guard_allows_valid():
    guard = SQLGuard({"users": ["id", "name", "age"]})
    sql, tables, cols = guard.check_sql("SELECT id, name FROM users WHERE age > 18")
    assert "users" in tables
    assert "id" in cols
    assert "limit 10000" in sql.lower()

def test_sql_guard_blocks_forbidden():
    guard = SQLGuard({"users": ["id", "name", "age"]})
    
    malicious_sqls = [
        "DROP TABLE users",
        "SELECT * FROM users; DROP TABLE users",
        "DELETE FROM users",
        "INSTALL postgres",
        "LOAD postgres",
        "COPY users TO 'out.csv'",
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "SELECT * FROM read_parquet('secret.parquet')",
        "SELECT * FROM glob('/*')",
        "SELECT * FROM sniff_csv('test.csv')",
        "PRAGMA table_info('users')",
        "SELECT * FROM system.information_schema.tables"
    ]
    
    for sql in malicious_sqls:
        with pytest.raises(SQLGuardError):
            guard.check_sql(sql)

def test_sql_guard_blocks_unauthorized_tables():
    guard = SQLGuard({"users": ["id", "name", "age"]})
    
    with pytest.raises(SQLGuardError, match="is not in the allowlist"):
        guard.check_sql("SELECT * FROM secrets")

def test_python_guard_allows_valid():
    guard = PythonGuard()
    code = """
import pandas as pd
import numpy as np

def compute():
    df = pd.DataFrame({"A": [1, 2]})
    return {"mean": df["A"].mean()}
"""
    guard.check_code(code)

def test_python_guard_blocks_forbidden():
    guard = PythonGuard()
    
    malicious_codes = [
        "import os\nos.system('rm -rf /')",
        "import subprocess",
        "import sys",
        "import pathlib",
        "eval('1+1')",
        "exec('a=1')",
        "compile('1+1', '', 'eval')",
        "__import__('os')",
        "open('/etc/passwd', 'r')",
        "import pandas as pd\npd.read_csv('/etc/passwd')",
        "import pandas as pd\ndf.to_csv('out.csv')",
        "import polars as pl\npl.read_parquet('secret.parquet')",
        "class A:\n def __getattr__(self, name):\n  pass\na=A()\na.__dict__"
    ]
    
    for code in malicious_codes:
        with pytest.raises(PythonGuardError):
            guard.check_code(code)
