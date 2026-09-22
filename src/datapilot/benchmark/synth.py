"""
Synthetic Data Generator with mathematical ground truth.
Generates realistic business scenarios with known treatment effects, confounders, and noise traps.
"""
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def generate_pricing_experiment(
    n_samples: int = 1200,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates a SaaS pricing change experiment dataset with Simpson's paradox confounding.

    - Segments: 'Enterprise' (20%), 'Mid-Market' (30%), 'SMB' (50%)
    - Confounder: Enterprise customers are more likely to adopt the new pricing (70%),
      while SMBs are less likely (25%).
    - Baseline revenue:
        Enterprise: ~$5000, Mid-Market: ~$1500, SMB: ~$200
    - True Treatment Effects:
        Enterprise: +12.0% ($600)
        Mid-Market: +8.0% ($120)
        SMB: 0.0% ($0)
    - Confounding trap: A naive comparison of treated vs untreated yields an inflated
      effect because treated is heavily skewed towards high-baseline Enterprise customers!
    - Noise columns: 'lucky_number', 'signup_hour' (pure noise, true effect = 0).
    """
    rng = np.random.default_rng(seed)

    # 1. Segments
    segments = rng.choice(
        ["Enterprise", "Mid-Market", "SMB"],
        size=n_samples,
        p=[0.20, 0.30, 0.50],
    )

    # 2. Confounded Treatment Assignment
    treatment_probs = {
        "Enterprise": 0.70,
        "Mid-Market": 0.50,
        "SMB": 0.25,
    }
    is_treated = np.array([rng.binomial(1, treatment_probs[s]) for s in segments])

    # 3. Pre-period baseline revenue
    base_means = {"Enterprise": 5000.0, "Mid-Market": 1500.0, "SMB": 200.0}
    base_stds = {"Enterprise": 800.0, "Mid-Market": 300.0, "SMB": 50.0}

    pre_revenue = np.array([
        max(10.0, rng.normal(base_means[s], base_stds[s]))
        for s in segments
    ])

    # 4. True Treatment Effects
    true_multipliers = {
        "Enterprise": 0.12,
        "Mid-Market": 0.08,
        "SMB": 0.00,
    }

    # Post-period revenue = pre_revenue * (1 + organic_noise) + (treatment_effect if treated)
    post_revenue = []
    for s, pre, trt in zip(segments, pre_revenue, is_treated):
        organic_growth = rng.normal(0.02, 0.04)  # 2% organic drift
        treatment_gain = (pre * true_multipliers[s]) if trt == 1 else 0.0
        post_val = pre * (1 + organic_growth) + treatment_gain
        post_revenue.append(max(10.0, post_val))

    post_revenue = np.array(post_revenue)

    # 5. Noise traps (for false discovery / p-hacking test)
    lucky_number = rng.integers(1, 100, size=n_samples)
    signup_hour = rng.integers(0, 24, size=n_samples)

    df = pd.DataFrame({
        "customer_id": [f"cust_{i:05d}" for i in range(n_samples)],
        "segment": segments,
        "is_treated": is_treated,
        "pre_revenue": np.round(pre_revenue, 2),
        "post_revenue": np.round(post_revenue, 2),
        "revenue_diff": np.round(post_revenue - pre_revenue, 2),
        "lucky_number": lucky_number,
        "signup_hour": signup_hour,
    })

    # True population weighted ATE in revenue gain
    treated_weights = [0.20 * 0.70, 0.30 * 0.50, 0.50 * 0.25]
    total_trt_weight = sum(treated_weights)
    true_effects_abs = [5000.0 * 0.12, 1500.0 * 0.08, 200.0 * 0.00]
    true_ate = sum(w * eff for w, eff in zip(treated_weights, true_effects_abs)) / total_trt_weight

    ground_truth = {
        "scenario": "saas_pricing_change",
        "true_ate_revenue_diff": round(float(true_ate), 2),
        "true_segment_multipliers": true_multipliers,
        "confounder_column": "segment",
        "treatment_column": "is_treated",
        "outcome_column": "revenue_diff",
        "noise_columns": ["lucky_number", "signup_hour"],
        "n_samples": n_samples,
    }

    return df, ground_truth
