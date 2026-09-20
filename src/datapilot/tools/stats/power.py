import statsmodels.stats.power as smp
from typing import Dict, Any

def power_analysis(effect_size: float, nobs1: float, alpha: float = 0.05, ratio: float = 1.0) -> Dict[str, Any]:
    """
    Computes statistical power for an independent t-test given the effect size and sample size.
    """
    # Initialize the power analysis object
    analysis = smp.TTestIndPower()
    
    # Calculate power
    try:
        power = analysis.solve_power(
            effect_size=abs(effect_size),
            nobs1=nobs1,
            alpha=alpha,
            ratio=ratio,
            power=None,
            alternative="two-sided"
        )
    except Exception:
        power = 0.0
        
    return {
        "effect_size": float(effect_size),
        "nobs1": float(nobs1),
        "ratio": float(ratio),
        "alpha": float(alpha),
        "power": float(power)
    }
