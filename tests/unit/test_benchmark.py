"""
Tests for Phase 8: Benchmark (Synthetic Data Generator, Naive Baseline, & Benchmark Runner).
"""
import pytest
from datapilot.benchmark.synth import generate_pricing_experiment
from datapilot.benchmark.baseline import NaiveBaselineAgent
from datapilot.benchmark.runner import BenchmarkRunner


def test_synthetic_pricing_experiment_structure():
    df, gt = generate_pricing_experiment(n_samples=500, seed=123)

    assert len(df) == 500
    expected_cols = {
        "customer_id", "segment", "is_treated", "pre_revenue",
        "post_revenue", "revenue_diff", "lucky_number", "signup_hour"
    }
    assert expected_cols.issubset(df.columns)

    # Check ground truth
    assert gt["scenario"] == "saas_pricing_change"
    assert gt["true_ate_revenue_diff"] > 0
    assert "Enterprise" in gt["true_segment_multipliers"]

    # Check reproducibility
    df2, gt2 = generate_pricing_experiment(n_samples=500, seed=123)
    assert df["revenue_diff"].sum() == df2["revenue_diff"].sum()
    assert gt["true_ate_revenue_diff"] == gt2["true_ate_revenue_diff"]


def test_naive_baseline_produces_confounded_estimate():
    df, gt = generate_pricing_experiment(n_samples=600, seed=42)
    baseline = NaiveBaselineAgent()
    res = baseline.analyze(df, "Evaluate pricing experiment")

    assert "estimated_ate" in res
    assert "conclusions" in res
    assert len(res["conclusions"]) >= 1
    assert res["has_provenance"] is False
    assert res["has_firewall"] is False


def test_benchmark_runner_executes_and_scores():
    runner = BenchmarkRunner()
    scoreboard = runner.run_benchmark(n_samples=800, seed=42)

    assert "scenario" in scoreboard
    assert "naive_baseline" in scoreboard
    assert "datapilot" in scoreboard

    nb = scoreboard["naive_baseline"]
    dp = scoreboard["datapilot"]

    # DataPilot should have 0 hallucinations, Naive baseline should have > 0
    assert dp["hallucination_rate"] == 0.0
    assert dp["firewall_violations"] == 0
    assert nb["hallucination_rate"] == 1.0
    assert nb["firewall_violations"] > 0

    # DataPilot controls for confounders
    assert dp["controlled_for_confounders"] is True
    assert nb["controlled_for_confounders"] is False

    # DataPilot should achieve lower ATE estimation error due to controlling for segment skew
    assert dp["ate_absolute_error"] < nb["ate_absolute_error"]

    # Test scoreboard formatting
    formatted = runner.format_scoreboard(scoreboard)
    assert "DATAPILOT BENCHMARK COMPARISON SCOREBOARD" in formatted
    assert "True Ground Truth ATE" in formatted
