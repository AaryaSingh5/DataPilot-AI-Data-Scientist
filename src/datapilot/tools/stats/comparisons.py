import numpy as np
from scipy import stats
import pingouin as pg
from typing import Dict, Any, List

def before_after(data_a: List[float], data_b: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    a = np.array(data_a)
    b = np.array(data_b)
    
    n_a, n_b = len(a), len(b)
    
    if n_a < 3 or n_b < 3:
        raise ValueError("Insufficient sample size")

    # Assumption checks
    # Shapiro-Wilk for normality if n < 5000, else we assume normality by CLT
    is_normal = True
    if max(n_a, n_b) < 5000:
        _, p_a = stats.shapiro(a)
        _, p_b = stats.shapiro(b)
        if p_a < alpha or p_b < alpha:
            is_normal = False
            
    # Levene's test for equal variances
    _, p_var = stats.levene(a, b)
    equal_var = p_var >= alpha

    if is_normal:
        # Welch's t-test (doesn't assume equal variance)
        test_res = pg.ttest(b, a, correction=not equal_var)
        test_name = "Welch's t-test" if not equal_var else "Student's t-test"
        p_val = float(test_res["p_val"].iloc[0])
        stat = float(test_res["T"].iloc[0])
        effect_size = float(test_res["cohen_d"].iloc[0])
        ci = test_res["CI95"].iloc[0]
    else:
        # Mann-Whitney U test
        test_res = pg.mwu(b, a)
        test_name = "Mann-Whitney U"
        p_val = float(test_res["p_val"].iloc[0])
        stat = float(test_res["U_val"].iloc[0])
        effect_size = float(test_res["RBC"].iloc[0])  # Rank-Biserial Correlation
        # Pingouin doesn't give CI for MWU easily, compute a simple bootstrap CI for median difference
        diffs = [np.median(np.random.choice(b, len(b))) - np.median(np.random.choice(a, len(a))) for _ in range(1000)]
        ci = [np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)]
        
    return {
        "test_name": test_name,
        "statistic": stat,
        "p": p_val,
        "effect_size": effect_size,
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b))
    }
