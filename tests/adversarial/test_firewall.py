"""
Adversarial tests for the 12-check Hallucination Firewall.
"""
import pytest
from datapilot.firewall.firewall import HallucinationFirewall


@pytest.fixture
def base_evidence():
    return [
        {
            "id": "ev_0001",
            "kind": "stat_test",
            "code": "before_after",
            "params": {},
            "status": "ok",
            "columns": ["revenue", "region"],
            "result": {
                "p": 0.003,
                "q": 0.008,
                "n": 100,
                "effect_size": 0.85,
                "diff": 15.5,
                "statistic": 3.42,
            },
        },
        {
            "id": "ev_0002",
            "kind": "model",
            "code": "gradient_boosting",
            "params": {"target": "churn"},
            "status": "ok",
            "columns": ["tenure", "usage", "churn"],
            "result": {
                "metrics": {"r2": 0.65},
                "feature_importance": {"tenure": 0.45, "usage": 0.55},
            },
        },
    ]


@pytest.fixture
def firewall():
    schema = {"sales": {"revenue": "DOUBLE", "region": "VARCHAR", "tenure": "INT", "usage": "DOUBLE", "churn": "INT"}}
    return HallucinationFirewall(schema=schema)


def test_firewall_allows_valid_claims(firewall, base_evidence):
    claims = [
        {
            "hypothesis_id": "H1",
            "verdict": "supported",
            "summary": "Revenue was significantly higher with an increase of 15.5 across 100 observations.",
            "evidence_ids": ["ev_0001"],
            "columns": ["revenue"],
        }
    ]
    res = firewall.verify_conclusions(claims, base_evidence)
    assert res.passed is True
    assert len(res.violations) == 0


def test_check_1_provenance_rejects_missing_or_fake_id(firewall, base_evidence):
    claims_no_id = [
        {
            "hypothesis_id": "H1",
            "verdict": "supported",
            "summary": "Revenue increased dramatically.",
            "evidence_ids": [],
        }
    ]
    res = firewall.verify_conclusions(claims_no_id, base_evidence)
    assert res.passed is False
    assert any(v.check_name == "provenance_grounding" for v in res.violations)

    claims_fake_id = [
        {
            "hypothesis_id": "H1",
            "verdict": "supported",
            "summary": "Revenue increased dramatically.",
            "evidence_ids": ["ev_9999"],
        }
    ]
    res = firewall.verify_conclusions(claims_fake_id, base_evidence)
    assert res.passed is False
    assert any("nonexistent" in v.message for v in res.violations)


def test_check_2_numerical_exactness_blocks_hallucinated_numbers(firewall, base_evidence):
    # Claim cites 88.5 which is nowhere in ev_0001
    claims = [
        {
            "hypothesis_id": "H1",
            "verdict": "supported",
            "summary": "Revenue grew by 88.5 units on average.",
            "evidence_ids": ["ev_0001"],
        }
    ]
    res = firewall.verify_conclusions(claims, base_evidence)
    assert res.passed is False
    assert any(v.check_name == "numerical_exactness" for v in res.violations)


def test_check_3_schema_integrity_blocks_fake_columns(firewall, base_evidence):
    # Claim references 'phantom_margin' not in schema
    claims = [
        {
            "hypothesis_id": "H1",
            "verdict": "supported",
            "summary": "Phantom margin was positive.",
            "evidence_ids": ["ev_0001"],
            "columns": ["phantom_margin"],
        }
    ]
    res = firewall.verify_conclusions(claims, base_evidence)
    assert res.passed is False
    assert any(v.check_name == "schema_integrity" for v in res.violations)


def test_check_4_causal_language_blocks_unjustified_causal_claims(firewall, base_evidence):
    # Claim asserts 'caused' on an observational before_after test without RCT
    claims = [
        {
            "hypothesis_id": "H1",
            "verdict": "supported",
            "summary": "The marketing change caused revenue to increase by 15.5.",
            "evidence_ids": ["ev_0001"],
        }
    ]
    res = firewall.verify_conclusions(claims, base_evidence)
    assert res.passed is False
    assert any(v.check_name == "causal_language_guard" for v in res.violations)


def test_check_5_significance_blocks_non_significant_claims(firewall):
    non_sig_evidence = [
        {
            "id": "ev_0010",
            "kind": "stat_test",
            "result": {"p": 0.35, "n": 50},
        }
    ]
    claims = [
        {
            "hypothesis_id": "H1",
            "verdict": "supported",
            "summary": "The price change had a statistically significant effect.",
            "evidence_ids": ["ev_0010"],
        }
    ]
    res = firewall.verify_conclusions(claims, non_sig_evidence)
    assert res.passed is False
    assert any(v.check_name == "significance_check" for v in res.violations)


def test_check_6_fdr_blocks_unjustified_claims_failing_q_threshold(firewall):
    fdr_failed_evidence = [
        {
            "id": "ev_0020",
            "kind": "stat_test",
            "result": {"p": 0.04, "q": 0.12, "n": 50},  # p < 0.05 but q > 0.05
        }
    ]
    claims = [
        {
            "hypothesis_id": "H1",
            "verdict": "supported",
            "summary": "We observed a statistically significant difference.",
            "evidence_ids": ["ev_0020"],
        }
    ]
    res = firewall.verify_conclusions(claims, fdr_failed_evidence)
    assert res.passed is False
    assert any(v.check_name == "fdr_check" for v in res.violations)


def test_check_7_effect_size_blocks_exaggerated_claims(firewall):
    tiny_effect_evidence = [
        {
            "id": "ev_0030",
            "kind": "stat_test",
            "result": {"p": 0.001, "effect_size": 0.08, "n": 500},
        }
    ]
    claims = [
        {
            "hypothesis_id": "H1",
            "verdict": "supported",
            "summary": "There is a strong effect across the cohort.",
            "evidence_ids": ["ev_0030"],
        }
    ]
    res = firewall.verify_conclusions(claims, tiny_effect_evidence)
    assert res.passed is False
    assert any(v.check_name == "effect_size_check" for v in res.violations)


def test_check_8_directionality_blocks_inverted_claims(firewall, base_evidence):
    # ev_0001 diff is +15.5, but claim says 'decreased'
    claims = [
        {
            "hypothesis_id": "H1",
            "verdict": "supported",
            "summary": "Revenue decreased after the experiment.",
            "evidence_ids": ["ev_0001"],
        }
    ]
    res = firewall.verify_conclusions(claims, base_evidence)
    assert res.passed is False
    assert any(v.check_name == "directionality_check" for v in res.violations)


def test_check_9_sample_size_blocks_fabricated_n(firewall, base_evidence):
    # ev_0001 has n=100, claim says n=9500
    claims = [
        {
            "hypothesis_id": "H1",
            "verdict": "supported",
            "summary": "Across 9500 users, revenue grew.",
            "evidence_ids": ["ev_0001"],
        }
    ]
    res = firewall.verify_conclusions(claims, base_evidence)
    assert res.passed is False
    assert any(v.check_name == "sample_size_check" for v in res.violations)


def test_check_10_leakage_blocks_target_in_model_features(firewall):
    leaky_evidence = [
        {
            "id": "ev_0040",
            "kind": "model",
            "params": {"target": "churn"},
            "result": {
                "feature_importance": {"churn": 0.98, "tenure": 0.02},
            },
        }
    ]
    claims = [
        {
            "hypothesis_id": "H1",
            "verdict": "supported",
            "summary": "Predictive model achieved high accuracy.",
            "evidence_ids": ["ev_0040"],
        }
    ]
    res = firewall.verify_conclusions(claims, leaky_evidence)
    assert res.passed is False
    assert any(v.check_name == "data_leakage_check" for v in res.violations)


def test_check_11_scope_warns_universal_claims_on_filtered_data(firewall):
    filtered_evidence = [
        {
            "id": "ev_0050",
            "kind": "sql",
            "code": "SELECT * FROM sales WHERE region = 'EU'",
            "params": {},
            "result": {"num_rows": 20},
        }
    ]
    claims = [
        {
            "hypothesis_id": "H1",
            "verdict": "supported",
            "summary": "The pattern holds globally across all segments.",
            "evidence_ids": ["ev_0050"],
        }
    ]
    res = firewall.verify_conclusions(claims, filtered_evidence)
    assert any(w.check_name == "scope_check" for w in res.warnings)


def test_check_12_non_vacuous_blocks_tautological_claims(firewall, base_evidence):
    claims = [
        {
            "hypothesis_id": "H1",
            "verdict": "supported",
            "summary": "Revenue is correlated with revenue because data exists.",
            "evidence_ids": ["ev_0001"],
        }
    ]
    res = firewall.verify_conclusions(claims, base_evidence)
    assert res.passed is False
    assert any(v.check_name == "non_vacuous_check" for v in res.violations)
