import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from typing import Dict, Any, List

def association(x: List[float], y: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        raise ValueError("Insufficient sample size")
        
    n = len(df)
    
    # Check for autocorrelation (proxy by checking diff correlation)
    x_diff = df["x"].diff().dropna()
    y_diff = df["y"].diff().dropna()
    
    # Pearson
    pearson = pg.corr(df["x"], df["y"], method="pearson")
    # Spearman
    spearman = pg.corr(df["x"], df["y"], method="spearman")
    
    # Diff correlation
    if len(x_diff) > 2:
        diff_corr = pg.corr(x_diff, y_diff, method="pearson")
        diff_r = float(diff_corr["r"].iloc[0])
        diff_p = float(diff_corr["p_val"].iloc[0])
    else:
        diff_r, diff_p = None, None
        
    return {
        "n": n,
        "pearson_r": float(pearson["r"].iloc[0]),
        "pearson_p": float(pearson["p_val"].iloc[0]),
        "pearson_ci_lower": float(pearson["CI95"].iloc[0][0]),
        "pearson_ci_upper": float(pearson["CI95"].iloc[0][1]),
        "spearman_r": float(spearman["r"].iloc[0]),
        "spearman_p": float(spearman["p_val"].iloc[0]),
        "diff_pearson_r": diff_r,
        "diff_pearson_p": diff_p
    }

def controlled_association(x: List[float], y: List[float], controls: Dict[str, List[float]]) -> Dict[str, Any]:
    data = {"x": x, "y": y}
    data.update(controls)
    df = pd.DataFrame(data).dropna()
    
    if len(df) < len(controls) + 3:
        raise ValueError("Insufficient sample size for controlled analysis")
        
    # Raw effect
    raw_corr = pg.corr(df["x"], df["y"])["r"].iloc[0]
    
    # Partial correlation
    covar = list(controls.keys())
    partial = pg.partial_corr(data=df, x="x", y="y", covar=covar)
    
    controlled_r = partial["r"].iloc[0]
    
    # OLS for VIF and HAC standard errors
    X = sm.add_constant(df[["x"] + covar])
    model = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    
    # VIF for x
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    x_vif = vifs[1] # Index 1 is "x" because 0 is constant
    
    attenuation = abs(controlled_r) / abs(raw_corr) if abs(raw_corr) > 1e-9 else 1.0
    
    return {
        "n": len(df),
        "raw_effect": float(raw_corr),
        "controlled_effect": float(controlled_r),
        "attenuation_ratio": float(attenuation),
        "p_value": float(partial["p-val"].iloc[0]),
        "ci_lower": float(partial["CI95%"].iloc[0][0]),
        "ci_upper": float(partial["CI95%"].iloc[0][1]),
        "vif_x": float(x_vif),
        "r_squared": float(model.rsquared)
    }
