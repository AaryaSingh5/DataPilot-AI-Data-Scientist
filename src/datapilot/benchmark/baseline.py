"""
Naive Baseline Agent — unconstrained baseline simulating typical unguarded AI data analysis.
Lacks FDR correction, causal controls, and Hallucination Firewall.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats


class NaiveBaselineAgent:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def analyze(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """
        Runs an unconstrained naive analysis on the dataset:
        1. Simple unadjusted comparison of treated vs untreated (falling for confounder bias).
        2. Unadjusted multiple correlations against noise columns (falling for p-hacking).
        3. Generates ungrounded, unverified causal claims with numbers.
        """
        # Naive treatment comparison (omits 'segment' confounder!)
        treated_rev = df[df["is_treated"] == 1]["revenue_diff"]
        control_rev = df[df["is_treated"] == 0]["revenue_diff"]

        t_stat, p_val = stats.ttest_ind(treated_rev, control_rev)
        naive_effect = float(treated_rev.mean() - control_rev.mean())

        # Unadjusted testing on noise columns (p-hacking)
        significant_noise = []
        for col in ["lucky_number", "signup_hour"]:
            if col in df.columns:
                r, p = stats.pearsonr(df[col], df["revenue_diff"])
                # Without FDR or Bonferroni, report whatever happens to have p < alpha
                if p < self.alpha:
                    significant_noise.append(col)

        # Naive causal claim without provenance, FDR, or confounder adjustment
        conclusions = [
            {
                "claim": f"The pricing experiment directly caused a massive revenue increase of ${naive_effect:.2f} per customer.",
                "is_causal": True,
                "claimed_effect": naive_effect,
                "p_value": float(p_val),
                "controlled_for_confounders": False,
                "evidence_ids": [],  # No evidence IDs (fails provenance)
            }
        ]

        if significant_noise:
            for col in significant_noise:
                conclusions.append({
                    "claim": f"User {col} is a statistically significant driver of revenue performance.",
                    "is_causal": True,
                    "claimed_effect": 0.0,
                    "p_value": 0.04,
                    "evidence_ids": [],
                })

        # Summary text with unverified statements and causal claims
        summary = (
            f"The pricing experiment was an unequivocal success. It directly caused an average revenue increase "
            f"of ${naive_effect:.2f} across all customer segments globally. "
            f"Our statistical analysis confirms p={p_val:.4e}."
        )

        return {
            "method": "naive_baseline",
            "estimated_ate": naive_effect,
            "p_value": float(p_val),
            "summary": summary,
            "conclusions": conclusions,
            "significant_noise_found": significant_noise,
            "has_provenance": False,
            "has_firewall": False,
        }
