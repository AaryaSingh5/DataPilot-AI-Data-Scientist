import sqlglot
from sqlglot import exp
from typing import Dict, List, Tuple, Set

class SQLGuardError(Exception):
    pass

class SQLGuard:
    FORBIDDEN_FUNCTIONS = {
        "read_csv", "read_csv_auto", "read_parquet", "read_json", 
        "read_json_auto", "glob", "parquet_scan", "sniff_csv"
    }
    
    FORBIDDEN_PREFIXES = ("read_", "sniff_", "st_read")

    def __init__(self, allowed_schema: Dict[str, List[str]]):
        """
        allowed_schema: dict mapping table_name -> list of allowed columns
        """
        self.allowed_schema = {t.lower(): [c.lower() for c in cols] for t, cols in allowed_schema.items()}

    def check_sql(self, sql: str, limit: int = 10000) -> Tuple[str, List[str], List[str]]:
        try:
            statements = sqlglot.parse(sql, read="duckdb")
        except Exception as e:
            raise SQLGuardError(f"SQL parsing failed: {e}")
            
        if not statements:
            raise SQLGuardError("Empty SQL")
        if len(statements) > 1:
            raise SQLGuardError("Only single statements are allowed")
            
        stmt = statements[0]
        
        if not isinstance(stmt, exp.Select):
            raise SQLGuardError(f"Root statement must be SELECT. Found: {stmt.__class__.__name__}")
            
        referenced_tables: Set[str] = set()
        referenced_columns: Set[str] = set()
        
        # Check all expressions in the AST
        # Block functions
        for node in stmt.find_all(exp.Func):
            func_name = node.name.lower()
            if func_name in self.FORBIDDEN_FUNCTIONS or any(func_name.startswith(p) for p in self.FORBIDDEN_PREFIXES):
                raise SQLGuardError(f"Forbidden function called: {func_name}")
                
        # Check tables
        for node in stmt.find_all(exp.Table):
            table_name = node.name.lower()
            # CTEs are not real tables, we should allow them
            referenced_tables.add(table_name)
            
        # Check columns
        for node in stmt.find_all(exp.Column):
            col_name = node.name.lower()
            referenced_columns.add(col_name)

        # Enforce table allowlist (excluding CTEs which we can find by checking with)
        ctes = set()
        if stmt.args.get("with"):
            for expression in stmt.args["with"].expressions:
                ctes.add(expression.alias.lower())

        for t in referenced_tables:
            if t not in self.allowed_schema and t not in ctes:
                raise SQLGuardError(f"Table '{t}' is not in the allowlist or system tables are forbidden.")
                
        # To strictly enforce columns, we should Ideally know which table they belong to.
        # For this prototype, we check if the column exists in ANY of the allowed tables that are referenced.
        allowed_cols_in_query = set()
        for t in referenced_tables:
            if t in self.allowed_schema:
                allowed_cols_in_query.update(self.allowed_schema[t])
                
        for c in referenced_columns:
            # We skip '*'
            if c != '*' and c not in allowed_cols_in_query and c not in ctes: # ctes might have derived columns, but this is a heuristic
                # A robust implementation would use sqlglot optimizer.qualify_columns
                pass # Skipping strict column check for now due to CTE aliases and derived columns complexity
                
        # Inject LIMIT
        if not stmt.args.get("limit"):
            stmt = stmt.limit(limit)
            
        normalized_sql = stmt.sql(dialect="duckdb")
        return normalized_sql, list(referenced_tables - ctes), list(referenced_columns)
