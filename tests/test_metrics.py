from voice_optimized_rag.utils.metrics import MetricsCollector


def test_metrics_report_percentiles_and_cache_status() -> None:
    metrics = MetricsCollector()
    for value in [10, 20, 30, 40]:
        metrics.record_latency("rag", "retrieval", value, cache_status="miss")
    metrics.record_latency("rag", "retrieval", 5, cache_status="hit")

    summary = metrics.summary()["latency"]["rag"]["retrieval"]

    assert summary["p50_ms"] == 20
    assert summary["p70_ms"] == 30
    assert summary["p100_ms"] == 40
    assert summary["cache_hit"]["count"] == 1
    assert summary["cache_miss"]["count"] == 4
