from observability.runtime import RuntimeObservability


def test_runtime_observability_summary():
    obs = RuntimeObservability(structured_logs=False)
    obs.record_backend_call(backend="groq", success=True, latency_ms=120.0, status_code=200)
    obs.record_backend_call(backend="groq", success=False, latency_ms=300.0, status_code=429, error_type="HTTPError")

    summary = obs.summary()

    assert summary["calls_total"] == 2
    assert summary["calls_success"] == 1
    assert summary["calls_failure"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["p95_latency_ms"] >= 120.0
    assert "perception_success_rate" in summary["slo_targets"]
    assert "perception_failure_ratio" in summary["error_budget_targets"]
