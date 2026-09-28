# DataPilot End-to-End Debugging Log

This document records symptoms, root causes, and fixes during end-to-end testing of DataPilot with real LLM backends (NVIDIA AI / Anthropic Claude).

---

## Log Entries

### Entry 1: Model Deprecation Error (NVIDIA Backend)
- **Symptom**: Execution failed at `Planner` node with `APIStatusError 410: The model 'meta/llama-3.3-70b-instruct' has reached its end of life`.
- **Root Cause**: The model `meta/llama-3.3-70b-instruct` was retired on NVIDIA's API endpoints.
- **Fix**: Updated default model in `NvidiaClient` to `meta/llama-3.2-11b-vision-instruct` (active, fast, multi-modal LLM endpoint).

### Entry 2: Evidence ID Numbers Flagged by Numerical Exactness Firewall Check
- **Symptom**: Firewall reported `[numerical_exactness] Claim H1: Number '4' in claim 'H1' cannot be verified against cited evidence values.`
- **Root Cause**: `_check_numerical_exactness` extracted digits from evidence ID references in the claim text (e.g. `ev_0004` -> `4`) and attempted to verify `4` as a statistical measurement float.
- **Fix**: Updated `_check_numerical_exactness` to ignore evidence ID string numbers (`ev_0*<num>`) alongside hypothesis ID numbers (`H<num>`).

### Entry 3: Non-Significant Result Verdict Mismatch
- **Symptom**: Firewall reported `[significance_check] Claim H2: Claim 'H2' claims statistical significance but cited test p-value is not < 0.05.`
- **Root Cause**: Prompt `ANALYST_SYSTEM` did not explicitly mandate setting `verdict: "inconclusive"` or `"refuted"` when statistical tests return `p >= 0.05`.
- **Fix**: Enhanced `ANALYST_SYSTEM` prompt rules explicitly requiring that non-significant tests (p >= 0.05) must produce `"inconclusive"` or `"refuted"` verdicts, never `"supported"`.

### Entry 4: Successful End-to-End Benchmark Execution
- **Symptom**: None (Full success).
- **Result**: Successfully executed `DataPilotWorkflow` on the synthetic pricing benchmark dataset using the `NvidiaClient` backend (`meta/llama-3.2-11b-vision-instruct`).
- **Verification**:
  - All 6 graph nodes executed without unhandled exceptions.
  - 12-check Hallucination Firewall passed with zero violations (`Firewall Passed: True`).
  - SHA-256 Hash Chained Ledger verified successfully (`Chain Verified: True`).
  - Reproducibility bundle ZIP generated and validated (`bundle_run_bench_8aba1e90.zip`, 5.7 KB).
