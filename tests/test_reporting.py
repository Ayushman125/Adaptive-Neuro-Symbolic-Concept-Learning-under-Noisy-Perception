import pytest

from evaluation import reporting


def test_mean_std_ci_single_value():
    mean, std, low, high = reporting.mean_std_ci([1.5])
    assert mean == 1.5
    assert std == 0.0
    assert low == 1.5
    assert high == 1.5


def test_aggregate_summary_produces_ci_fields():
    rows = [
        {"benchmark": "b", "ablation": "full", "accuracy": 0.8, "brier": 0.2, "convergence_step": 3,
         "entropy_slope": -0.1, "correction_count": 2},
        {"benchmark": "b", "ablation": "full", "accuracy": 0.9, "brier": 0.1, "convergence_step": 4,
         "entropy_slope": -0.2, "correction_count": 3},
    ]
    summary = reporting.aggregate_summary(rows)
    assert summary
    item = summary[0]
    assert "accuracy_ci95_low" in item
    assert "accuracy_ci95_high" in item


def test_build_ablation_deltas():
    summary = [
        {
            "benchmark": "b",
            "ablation": "full",
            "accuracy_mean": 0.9,
            "brier_mean": 0.1,
            "convergence_step_mean": 3,
            "entropy_slope_mean": -0.1,
            "correction_count_mean": 2,
        },
        {
            "benchmark": "b",
            "ablation": "ablate_recency_blend",
            "accuracy_mean": 0.8,
            "brier_mean": 0.2,
            "convergence_step_mean": 4,
            "entropy_slope_mean": -0.2,
            "correction_count_mean": 3,
        },
    ]
    variants = {"full": {}, "ablate_recency_blend": {"recency_blend": False}}
    deltas = reporting.build_ablation_deltas(summary, variants)
    assert deltas
    assert deltas[0]["delta_accuracy"] == pytest.approx(-0.1, rel=1e-9)
