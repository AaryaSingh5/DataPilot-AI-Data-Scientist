"""
Hallucination Firewall — 12-check verification engine for all generated claims.
Intercepts Analyst and Report claims before publication.
"""
import re
import math
from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel


class FirewallViolation(BaseModel):
    check_name: str
    claim_id: Optional[str] = None
    message: str
    severity: str = "critical"  # "critical" or "warning"


class FirewallResult(BaseModel):
    passed: bool
    violations: List[FirewallViolation] = []
    warnings: List[FirewallViolation] = []

    def summary(self) -> str:
        if self.passed:
            return "Firewall Passed: All 12 checks satisfied."
        return f"Firewall Failed: {len(self.violations)} critical violations detected."


class HallucinationFirewall:
    CAUSAL_KEYWORDS = [
        r"\bcaused\b",
        r"\bcauses\b",
        r"\bcausing\b",
        r"\bproves\b",
        r"\bdirectly drives\b",
        r"\bdrove\b",
        r"\bled to\b",
        r"\bresults in\b",
    ]

    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        alpha: float = 0.05,
        fdr_q: float = 0.05,
        relative_tolerance: float = 0.02,
    ):
        self.schema = schema or {}
        self.alpha = alpha
        self.fdr_q = fdr_q
        self.relative_tolerance = relative_tolerance

    def verify_conclusions(
        self,
        conclusions: List[Dict[str, Any]],
        evidences: List[Dict[str, Any]],
    ) -> FirewallResult:
        ev_map = {e.get("id"): e for e in evidences if e.get("id")}
        violations: List[FirewallViolation] = []
        warnings: List[FirewallViolation] = []

        for idx, claim in enumerate(conclusions):
            cid = claim.get("hypothesis_id", f"claim_{idx+1}")
            c_ev_ids = claim.get("evidence_ids", [])
            text = (claim.get("summary", "") + " " + claim.get("verdict", "")).strip()

            # Check 1: Provenance Grounding
            self._check_provenance(cid, c_ev_ids, ev_map, violations)

            # Retrieve evidence dicts for this claim
            c_evs = [ev_map[eid] for eid in c_ev_ids if eid in ev_map]

            # Check 2: Numerical Exactness
            self._check_numerical_exactness(cid, text, c_evs, violations)

            # Check 3: Schema / Column Integrity
            self._check_schema_integrity(cid, text, claim, violations)

            # Check 4: Causal Language Guard
            self._check_causal_language(cid, text, c_evs, violations)

            # Check 5: Statistical Significance Check
            self._check_significance(cid, text, c_evs, violations)

            # Check 6: Multiple Testing Correction (FDR) Check
            self._check_fdr(cid, text, c_evs, violations)

            # Check 7: Effect Size Consistency
            self._check_effect_size(cid, text, c_evs, violations)

            # Check 8: Directionality Check
            self._check_directionality(cid, text, c_evs, violations)

            # Check 9: Sample Size Consistency
            self._check_sample_size(cid, text, c_evs, violations)

            # Check 10: Data Leakage Check
            self._check_data_leakage(cid, c_evs, violations)

            # Check 11: Scope / Subgroup Check
            self._check_scope(cid, text, c_evs, warnings)

            # Check 12: Non-Vacuous Claims Check
            self._check_non_vacuous(cid, text, violations)

        passed = len(violations) == 0
        return FirewallResult(passed=passed, violations=violations, warnings=warnings)

    # 1. Provenance
    def _check_provenance(self, cid: str, ev_ids: List[str], ev_map: Dict[str, Any], violations: List[FirewallViolation]):
        if not ev_ids:
            violations.append(FirewallViolation(
                check_name="provenance_grounding",
                claim_id=cid,
                message=f"Claim '{cid}' does not cite any evidence IDs.",
            ))
            return

        for eid in ev_ids:
            if eid not in ev_map:
                violations.append(FirewallViolation(
                    check_name="provenance_grounding",
                    claim_id=cid,
                    message=f"Claim '{cid}' cites nonexistent evidence ID '{eid}'.",
                ))

    # 2. Numerical Exactness
    def _check_numerical_exactness(self, cid: str, text: str, evs: List[Dict[str, Any]], violations: List[FirewallViolation]):
        numbers_in_text = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
        clean_nums = []
        for n_str in numbers_in_text:
            is_pct = n_str.endswith("%")
            raw_val = n_str.rstrip("%")
            try:
                val = float(raw_val)
                # Ignore trivial years or index numbers like 1, 2 or common single digit integers if not metric
                clean_nums.append((val, is_pct, n_str))
            except ValueError:
                continue

        if not clean_nums or not evs:
            return

        # Collect all numeric values in evidence results
        ev_nums = set()
        for ev in evs:
            res = ev.get("result", {})
            self._extract_numbers(res, ev_nums)

        for val, is_pct, raw_str in clean_nums:
            # Skip hypothesis numbers (e.g. H1 -> 1)
            if re.search(rf"\b[Hh]{int(val)}\b", text):
                continue
            matched = False
            candidates = [val]
            if is_pct:
                candidates.append(val / 100.0)
            else:
                candidates.append(val * 100.0)

            for target in ev_nums:
                for c in candidates:
                    tol = max(abs(target) * self.relative_tolerance, 0.01)
                    if abs(c - target) <= tol:
                        matched = True
                        break
                if matched:
                    break

            if not matched and val not in (0.0, 1.0):
                violations.append(FirewallViolation(
                    check_name="numerical_exactness",
                    claim_id=cid,
                    message=f"Number '{raw_str}' in claim '{cid}' cannot be verified against cited evidence values.",
                ))

    def _extract_numbers(self, obj: Any, out: Set[float]):
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            if math.isfinite(obj):
                out.add(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                self._extract_numbers(v, out)
        elif isinstance(obj, list):
            for v in obj:
                self._extract_numbers(v, out)

    # 3. Schema Integrity
    def _check_schema_integrity(self, cid: str, text: str, claim: Dict[str, Any], violations: List[FirewallViolation]):
        if not self.schema:
            return
        valid_cols = set()
        for k, v in self.schema.items():
            if isinstance(v, dict):
                valid_cols.update(v.keys())
            else:
                valid_cols.add(k)

        # Check explicit columns list in claim if provided
        cols_mentioned = claim.get("columns", [])
        for col in cols_mentioned:
            if col not in valid_cols:
                violations.append(FirewallViolation(
                    check_name="schema_integrity",
                    claim_id=cid,
                    message=f"Claim '{cid}' references column '{col}' not present in dataset schema.",
                ))

    # 4. Causal Language Guard
    def _check_causal_language(self, cid: str, text: str, evs: List[Dict[str, Any]], violations: List[FirewallViolation]):
        has_causal_word = any(re.search(pat, text, re.IGNORECASE) for pat in self.CAUSAL_KEYWORDS)
        if not has_causal_word:
            return

        is_causally_justified = False
        for ev in evs:
            code = ev.get("code", "")
            params = ev.get("params", {})
            # Causal methods: controlled_association with low VIF and high attenuation, or RCT/DiD
            if code == "controlled_association":
                res = ev.get("result", {})
                if res.get("attenuation_ratio", 0) > 0.7 and res.get("vif_x", 99) < 5.0:
                    is_causally_justified = True
            elif params.get("is_rct") or "difference_in_differences" in code:
                is_causally_justified = True

        if not is_causally_justified:
            violations.append(FirewallViolation(
                check_name="causal_language_guard",
                claim_id=cid,
                message=f"Claim '{cid}' uses causal terminology without causal experimental or controlled design.",
            ))

    # 5. Significance Check
    def _check_significance(self, cid: str, text: str, evs: List[Dict[str, Any]], violations: List[FirewallViolation]):
        claims_significant = bool(re.search(r"\bsignificant(?:ly)?\b", text, re.IGNORECASE))
        if not claims_significant:
            return

        has_sig_test = False
        for ev in evs:
            if ev.get("kind") == "stat_test":
                res = ev.get("result", {})
                p = res.get("p") or res.get("p_value") or res.get("pearson_p") or res.get("spearman_p")
                if p is not None and p < self.alpha:
                    has_sig_test = True

        if not has_sig_test:
            violations.append(FirewallViolation(
                check_name="significance_check",
                claim_id=cid,
                message=f"Claim '{cid}' claims statistical significance but cited test p-value is not < {self.alpha}.",
            ))

    # 6. FDR Check
    def _check_fdr(self, cid: str, text: str, evs: List[Dict[str, Any]], violations: List[FirewallViolation]):
        claims_significant = bool(re.search(r"\bsignificant(?:ly)?\b", text, re.IGNORECASE))
        if not claims_significant:
            return

        for ev in evs:
            if ev.get("kind") == "stat_test":
                res = ev.get("result", {})
                q = res.get("q") or res.get("q_val")
                if q is not None and q >= self.fdr_q:
                    violations.append(FirewallViolation(
                        check_name="fdr_check",
                        claim_id=cid,
                        message=f"Claim '{cid}' claims significance but test fails FDR multiple testing threshold (q={q:.4f} >= {self.fdr_q}).",
                    ))

    # 7. Effect Size Check
    def _check_effect_size(self, cid: str, text: str, evs: List[Dict[str, Any]], violations: List[FirewallViolation]):
        claims_large = bool(re.search(r"\b(?:large|strong|substantial|huge)\s+(?:effect|correlation|impact|relationship)\b", text, re.IGNORECASE))
        if not claims_large:
            return

        supported = False
        for ev in evs:
            res = ev.get("result", {})
            d = res.get("effect_size")
            r = res.get("pearson_r") or res.get("raw_effect") or res.get("spearman_r")
            if d is not None and abs(d) >= 0.8:
                supported = True
            if r is not None and abs(r) >= 0.5:
                supported = True

        if not supported:
            violations.append(FirewallViolation(
                check_name="effect_size_check",
                claim_id=cid,
                message=f"Claim '{cid}' asserts a 'large/strong' effect, but evidence effect size does not meet large thresholds (|d|>=0.8 or |r|>=0.5).",
            ))

    # 8. Directionality Check
    def _check_directionality(self, cid: str, text: str, evs: List[Dict[str, Any]], violations: List[FirewallViolation]):
        claims_positive = bool(re.search(r"\b(?:increased|higher|gain|growth|rose|positive)\b", text, re.IGNORECASE))
        claims_negative = bool(re.search(r"\b(?:decreased|lower|loss|dropped|fell|negative)\b", text, re.IGNORECASE))

        if claims_positive == claims_negative:
            return  # Ambiguous or both mentioned

        for ev in evs:
            res = ev.get("result", {})
            stat = res.get("statistic") or res.get("pearson_r") or res.get("raw_effect")
            diff = res.get("diff")
            if diff is None and "mean_b" in res and "mean_a" in res:
                diff = res["mean_b"] - res["mean_a"]

            metric = diff if diff is not None else stat
            if metric is not None:
                if claims_positive and metric < -1e-6:
                    violations.append(FirewallViolation(
                        check_name="directionality_check",
                        claim_id=cid,
                        message=f"Claim '{cid}' asserts positive increase, but metric is negative ({metric:.4f}).",
                    ))
                elif claims_negative and metric > 1e-6:
                    violations.append(FirewallViolation(
                        check_name="directionality_check",
                        claim_id=cid,
                        message=f"Claim '{cid}' asserts negative decrease, but metric is positive ({metric:.4f}).",
                    ))

    # 9. Sample Size Check
    def _check_sample_size(self, cid: str, text: str, evs: List[Dict[str, Any]], violations: List[FirewallViolation]):
        n_match = re.search(r"\b(?:n\s*=\s*|sample\s+(?:size\s+)?(?:of\s+)?|across\s+)(\d+)\b", text, re.IGNORECASE)
        if not n_match:
            return

        claimed_n = int(n_match.group(1))
        found_match = False
        for ev in evs:
            res = ev.get("result", {})
            actual_n = res.get("n") or res.get("num_rows") or res.get("n_a")
            if actual_n is not None and abs(claimed_n - actual_n) <= max(1, actual_n * 0.05):
                found_match = True
                break

        if not found_match:
            violations.append(FirewallViolation(
                check_name="sample_size_check",
                claim_id=cid,
                message=f"Claim '{cid}' claims sample size N={claimed_n} which does not match cited evidence N.",
            ))

    # 10. Data Leakage Check
    def _check_data_leakage(self, cid: str, evs: List[Dict[str, Any]], violations: List[FirewallViolation]):
        for ev in evs:
            if ev.get("kind") == "model":
                params = ev.get("params", {})
                target = params.get("target")
                res = ev.get("result", {})
                fi = res.get("feature_importance", {})
                if target and target in fi:
                    violations.append(FirewallViolation(
                        check_name="data_leakage_check",
                        claim_id=cid,
                        message=f"Model in claim '{cid}' has target column '{target}' present in feature importances (leakage).",
                    ))

    # 11. Scope Check
    def _check_scope(self, cid: str, text: str, evs: List[Dict[str, Any]], warnings: List[FirewallViolation]):
        # If text makes universal claims like "across all segments" or "globally"
        if re.search(r"\bacross all\b|\bglobally\b|\buniversal\b", text, re.IGNORECASE):
            for ev in evs:
                code = ev.get("code", "")
                if "WHERE" in code.upper() or "filter" in str(ev.get("params", {})).lower():
                    warnings.append(FirewallViolation(
                        check_name="scope_check",
                        claim_id=cid,
                        message=f"Claim '{cid}' claims universal scope, but evidence used filtered data.",
                        severity="warning",
                    ))

    # 12. Non-Vacuous Check
    def _check_non_vacuous(self, cid: str, text: str, violations: List[FirewallViolation]):
        words = text.split()
        if len(words) < 3:
            violations.append(FirewallViolation(
                check_name="non_vacuous_check",
                claim_id=cid,
                message=f"Claim '{cid}' is trivial or empty ({len(words)} words).",
            ))
            return

        tautology_patterns = [
            r"revenue is correlated with revenue",
            r"data exists",
            r"nothing to report",
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in tautology_patterns):
            violations.append(FirewallViolation(
                check_name="non_vacuous_check",
                claim_id=cid,
                message=f"Claim '{cid}' is tautological or content-free.",
            ))
