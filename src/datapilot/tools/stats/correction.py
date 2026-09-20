import numpy as np
from typing import List, Dict, Any

def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """
    Computes Benjamini-Hochberg FDR corrected q-values.
    """
    n = len(p_values)
    if n == 0:
        return []
        
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # Calculate q-values: p * n / rank
    ranks = np.arange(1, n + 1)
    q_values = sorted_p * n / ranks
    
    # Ensure q-values are monotonically increasing (enforce q_i = min(q_i, q_i+1))
    for i in range(n - 2, -1, -1):
        q_values[i] = min(q_values[i], q_values[i + 1])
        
    # Cap at 1.0
    q_values = np.minimum(q_values, 1.0)
    
    # Reorder back to original
    original_order_q = np.zeros(n)
    original_order_q[sorted_indices] = q_values
    
    return original_order_q.tolist()

def apply_fdr_correction(tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Extracts p-values, runs BH, injects q-values back
    p_vals = []
    for t in tests:
        # Some tests might not have p (e.g. if failed)
        p = t.get("result", {}).get("p", t.get("result", {}).get("p_value"))
        p_vals.append(p if p is not None else 1.0)
        
    q_vals = benjamini_hochberg(p_vals)
    
    for i, t in enumerate(tests):
        if "result" in t and isinstance(t["result"], dict):
            t["result"]["q_value"] = q_vals[i]
            t["result"]["family_size"] = len(p_vals)
            
    return tests
