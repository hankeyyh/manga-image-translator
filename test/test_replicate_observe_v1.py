from replicate.observe_v1 import phase2_recommendation, summarize_predictions


def _prediction(status: str, *, error: str | None = None, duration_sec: int = 10):
    return {
        "status": status,
        "error": error,
        "started_at": "2026-01-01T00:00:00Z",
        "completed_at": f"2026-01-01T00:00:{duration_sec:02d}Z",
    }


def test_summarize_predictions_counts_timeout_and_oom():
    predictions = [
        _prediction("succeeded", duration_sec=6),
        _prediction("failed", error="CUDA out of memory"),
        _prediction("failed", error="request timeout after 60s"),
        _prediction("processing"),
    ]

    summary = summarize_predictions(predictions)
    assert summary.total == 4
    assert summary.succeeded == 1
    assert summary.failed == 2
    assert summary.pending_or_running == 1
    assert summary.oom_failures == 1
    assert summary.timeout_failures == 1
    assert summary.avg_latency_sec == 6.0
    assert summary.p95_latency_sec == 6.0


def test_phase2_recommendation_requires_enough_samples():
    predictions = [_prediction("succeeded", duration_sec=5) for _ in range(10)]
    summary = summarize_predictions(predictions)
    assert "样本量不足" in phase2_recommendation(summary)


def test_phase2_recommendation_reports_oom_risk():
    predictions = [_prediction("succeeded", duration_sec=5) for _ in range(98)] + [
        _prediction("failed", error="OOM killed"),
        _prediction("failed", error="cuda out of memory"),
    ]
    summary = summarize_predictions(predictions)
    assert "OOM 率偏高" in phase2_recommendation(summary)

