import ast
from typing import Set

class PythonGuardError(Exception):
    pass

class PythonGuard(ast.NodeVisitor):
    ALLOWED_MODULES = {
        "pandas", "polars", "numpy", "scipy", "statsmodels", 
        "sklearn", "plotly", "math", "statistics", "datetime", "json"
    }
    
    FORBIDDEN_CALLS = {
        "eval", "exec", "compile", "__import__", "open",
        # Pandas/Polars specific bans
        "read_csv", "read_parquet", "read_json", "read_excel", "read_sql",
        "to_csv", "to_parquet", "to_json", "to_excel", "to_sql"
    }
    
    def __init__(self):
        self.errors = []

    def check_code(self, code: str) -> None:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise PythonGuardError(f"Syntax Error: {e}")
            
        self.visit(tree)
        if self.errors:
            raise PythonGuardError(" | ".join(self.errors))

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_module = alias.name.split('.')[0]
            if base_module not in self.ALLOWED_MODULES:
                self.errors.append(f"Importing '{alias.name}' is not allowed.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_module = node.module.split('.')[0]
            if base_module not in self.ALLOWED_MODULES:
                self.errors.append(f"Importing from '{node.module}' is not allowed.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.FORBIDDEN_CALLS:
                self.errors.append(f"Function call '{node.func.id}' is forbidden.")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.FORBIDDEN_CALLS:
                self.errors.append(f"Method call '{node.func.attr}' is forbidden.")
                
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.errors.append(f"Dunder attribute access '{node.attr}' is forbidden.")
        self.generic_visit(node)
