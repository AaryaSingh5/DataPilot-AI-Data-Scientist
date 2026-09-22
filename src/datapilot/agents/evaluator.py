"""
Evaluator Agent — assesses evidence against statistical rigor thresholds (thresholds.yaml).
"""
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datapilot.ledger.store import LedgerStore


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"


class EvaluatorAgent:
    def __init__(self, store: LedgerStore, config_path: Optional[Path] = None):
        self.store = store
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.thresholds = self._load_thresholds(self.config_path)

    def _load_thresholds(self, path: Path) -> Dict[str, Any]:
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {
            "significance": {"alpha": 0.05, "fdr_q": 0.05, "high_confidence_alpha": 0.01},
            "effect_size": {
                "cohens_d": {"small": 0.2, "medium": 0.5, "large": 0.8},
                "pearson_r": {"small": 0.1, "medium": 0.3, "large": 0.5},
            },
            "sample_size": {"min_n": 30, "warning_n": 50},
            "power": {"target_power": 0.80, "alpha": 0.05},
            "collinearity": {"max_vif": 5.0, "extreme_vif": 10.0},
            "machine_learning": {"min_r2": 0.0, "leakage_feature_importance_threshold": 0.95},
        }

    def evaluate_evidence(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        kind = ev.get("kind")
        result = ev.get("result", {})
        issues = []
        status = "pass"

        if ev.get("status") == "error":
            return {
                "evidence_id": ev.get("id"),
                "status": "failed",
                "score": 0.0,
                "issues": [f"Execution error: {ev.get('error')}"],
            }

        if kind == "stat_test":
            sig = self.thresholds.get("significance", {})
            ss = self.thresholds.get("sample_size", {})
            collin = self.thresholds.get("collinearity", {})
            eff = self.thresholds.get("effect_size", {})

            # Sample size check
            n = result.get("n") or result.get("n_a")
            if n is not None:
                if n < ss.get("min_n", 30):
                    issues.append(f"Sample size {n} is below minimum recommended {ss.get('min_n', 30)}")
                    status = "warning"
                elif n < ss.get("warning_n", 50):
                    issues.append(f"Sample size {n} is small (< {ss.get('warning_n', 50)})")
                    if status != "failed":
                        status = "warning"

            # p-value check
            p = result.get("p") or result.get("p_value") or result.get("pearson_p") or result.get("spearman_p")
            q = result.get("q") or result.get("q_val")
            alpha = sig.get("alpha", 0.05)
            fdr_q = sig.get("fdr_q", 0.05)

            if p is not None and p > alpha:
                issues.append(f"Test is not statistically significant (p = {p:.4f} > {alpha})")
                if status == "pass":
                    status = "warning"
            if q is not None and q > fdr_q:
                issues.append(f"Test does not survive FDR multiple testing correction (q = {q:.4f} > {fdr_q})")
                if status == "pass":
                    status = "warning"

            # Collinearity check
            vif = result.get("vif_x")
            if vif is not None and vif > collin.get("max_vif", 5.0):
                issues.append(f"High multicollinearity detected (VIF = {vif:.2f} > {collin.get('max_vif', 5.0)})")
                status = "warning"

            # Effect size check
            d = result.get("effect_size")
            if d is not None and isinstance(d, (int, float)):
                d_abs = abs(d)
                small_d = eff.get("cohens_d", {}).get("small", 0.2)
                if d_abs < small_d:
                    issues.append(f"Negligible effect size (|d| = {d_abs:.3f} < {small_d})")

        elif kind == "model":
            ml_th = self.thresholds.get("machine_learning", {})
            metrics = result.get("metrics", {})
            r2 = metrics.get("r2")
            if r2 is not None and r2 < ml_th.get("min_r2", 0.0):
                issues.append(f"Model R² ({r2:.3f}) is negative or below baseline threshold")
                status = "warning"

            # Leakage check in feature importance
            fi = result.get("feature_importance", {})
            leak_th = ml_th.get("leakage_feature_importance_threshold", 0.95)
            for feat, imp in fi.items():
                if imp >= leak_th:
                    issues.append(f"Possible data leakage: feature '{feat}' has {imp*100:.1f}% importance")
                    status = "warning"

        elif kind == "sql":
            num_rows = result.get("num_rows", 0)
            if num_rows == 0:
                issues.append("SQL query returned 0 rows")
                status = "warning"

        score = 1.0 if status == "pass" else (0.7 if status == "warning" else 0.0)
        return {
            "evidence_id": ev.get("id"),
            "status": status,
            "score": score,
            "issues": issues,
        }

    def run(
        self,
        run_id: str,
        dataset_hash: str,
        evidences: List[Dict[str, Any]],
        depends_on: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        evaluations = [self.evaluate_evidence(e) for e in evidences]
        total_score = sum(e["score"] for e in evaluations) / max(len(evaluations), 1)
        all_issues = [issue for e in evaluations for issue in e["issues"]]
        overall_status = "passed" if total_score >= 0.85 else ("warning" if total_score >= 0.5 else "failed")

        eval_result = {
            "overall_status": overall_status,
            "overall_score": round(total_score, 3),
            "evaluations": evaluations,
            "issues": all_issues,
            "total_evidences": len(evidences),
        }

        ev = self.store.record_evidence(
            run_id=run_id,
            kind="evaluation",
            produced_by="evaluator",
            dataset_hash=dataset_hash,
            code="evaluator_threshold_checks",
            params={"thresholds": self.thresholds},
            result=eval_result,
            columns=[],
            status="ok",
            depends_on=depends_on,
        )
        eval_result["_evidence_id"] = ev.id
        return eval_result
