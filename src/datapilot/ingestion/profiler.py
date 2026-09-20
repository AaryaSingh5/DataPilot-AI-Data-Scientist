import duckdb
from typing import Dict, Any, List

def profile_table(conn: duckdb.DuckDBPyConnection, table_name: str) -> Dict[str, Dict[str, Any]]:
    # Get total row count
    row_count = conn.execute(f"SELECT COUNT(*) FROM \"{table_name}\"").fetchone()[0]
    if row_count == 0:
        return {}
        
    schema_info = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    
    profile = {}
    for col in schema_info:
        col_name = col[1]
        col_type = col[2]
        
        # Get basic stats
        stats_query = f"""
            SELECT 
                COUNT("{col_name}") as non_null_count,
                COUNT(DISTINCT "{col_name}") as distinct_count,
                MIN("{col_name}") as min_val,
                MAX("{col_name}") as max_val
            FROM "{table_name}"
        """
        stats = conn.execute(stats_query).fetchone()
        non_null_count, distinct_count, min_val, max_val = stats
        
        null_rate = (row_count - non_null_count) / row_count
        
        # Get sample values (non-null, distinct)
        sample_query = f"""
            SELECT DISTINCT "{col_name}" 
            FROM "{table_name}" 
            WHERE "{col_name}" IS NOT NULL 
            LIMIT 5
        """
        samples = [row[0] for row in conn.execute(sample_query).fetchall()]
        
        # Sanitize samples (convert to string, limit length)
        sanitized_samples = []
        for s in samples:
            s_str = str(s).strip()
            # Basic control char stripping could go here
            if len(s_str) > 40:
                s_str = s_str[:37] + "..."
            sanitized_samples.append(s_str)
            
        # Semantic type heuristic
        t_lower = col_type.lower()
        semantic = "text"
        if "date" in t_lower or "time" in t_lower:
            semantic = "date"
        elif "int" in t_lower or "float" in t_lower or "double" in t_lower or "decimal" in t_lower or "numeric" in t_lower:
            if distinct_count <= 2:
                semantic = "boolean"
            elif col_name.lower().endswith("id") or col_name.lower() == "id":
                semantic = "id"
            else:
                semantic = "metric"
        elif "bool" in t_lower:
            semantic = "boolean"
        else:
            if distinct_count < 100 or distinct_count / row_count < 0.05:
                semantic = "category"
            elif col_name.lower().endswith("id") or col_name.lower() == "id":
                semantic = "id"

        profile[col_name] = {
            "type": col_type,
            "null_rate": float(null_rate),
            "distinct_count": int(distinct_count),
            "min": str(min_val) if min_val is not None else None,
            "max": str(max_val) if max_val is not None else None,
            "samples": sanitized_samples,
            "semantic_type": semantic
        }
        
    return profile
