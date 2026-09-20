import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Any, List

def analyze_timeseries(values: List[float], dates: List[str] = None, period: int = 12) -> Dict[str, Any]:
    s = pd.Series(values).dropna()
    if len(s) < period * 2:
        raise ValueError("Insufficient data for time series analysis")
        
    # Stationarity (ADF test)
    adf_res = adfuller(s)
    adf_p = adf_res[1]
    is_stationary = adf_p < 0.05
    
    # STL Decomposition
    try:
        stl = STL(s, period=period, robust=True).fit()
        var_seasonal = np.var(stl.seasonal)
        var_resid = np.var(stl.resid)
        # Seasonal strength: max(0, 1 - var(resid)/var(seasonal + resid))
        # Simplified:
        var_sr = np.var(stl.seasonal + stl.resid)
        seasonal_strength = max(0, 1 - (var_resid / var_sr)) if var_sr > 0 else 0
    except Exception:
        seasonal_strength = 0.0
        
    return {
        "n": len(s),
        "is_stationary": bool(is_stationary),
        "adf_p_value": float(adf_p),
        "seasonal_strength": float(seasonal_strength)
    }
