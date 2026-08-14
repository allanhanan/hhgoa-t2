"""Latency instrumentation and metrics tracking."""

from __future__ import annotations

import time
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyRecord:
    """A single latency measurement."""
    component: str
    operation: str
    duration_ms: float
    timestamp: float
    cache_status: str | None = None


class MetricsCollector:
    """Collects and reports latency metrics for all system components."""

    def __init__(self) -> None:
        self._records: list[LatencyRecord] = []
        self._counters: dict[str, int] = defaultdict(int)

    def record_latency(
        self,
        component: str,
        operation: str,
        duration_ms: float,
        cache_status: str | None = None,
    ) -> None:
        self._records.append(LatencyRecord(
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            timestamp=time.time(),
            cache_status=cache_status,
        ))

    def increment(self, counter_name: str, amount: int = 1) -> None:
        self._counters[counter_name] += amount

    def get_counter(self, counter_name: str) -> int:
        return self._counters.get(counter_name, 0)

    @property
    def cache_hit_rate(self) -> float:
        hits = self._counters.get("cache_hit", 0)
        misses = self._counters.get("cache_miss", 0)
        total = hits + misses
        return hits / total if total > 0 else 0.0

    def get_avg_latency(self, component: str, operation: str | None = None) -> float:
        filtered = [
            r for r in self._records
            if r.component == component
            and (operation is None or r.operation == operation)
        ]
        if not filtered:
            return 0.0
        return sum(r.duration_ms for r in filtered) / len(filtered)

    def get_percentile_latency(
        self,
        component: str,
        operation: str | None = None,
        percentile: float = 99,
        cache_status: str | None = None,
    ) -> float:
        filtered = sorted(
            (r.duration_ms for r in self._records
             if r.component == component
             and (operation is None or r.operation == operation)
             and (cache_status is None or r.cache_status == cache_status))
        )
        if not filtered:
            return 0.0
        idx = max(0, math.ceil(len(filtered) * (percentile / 100)) - 1)
        return filtered[min(idx, len(filtered) - 1)]

    def get_p99_latency(self, component: str, operation: str | None = None) -> float:
        return self.get_percentile_latency(component, operation, 99)

    def stage_summary(
        self,
        component: str,
        operation: str,
        cache_status: str | None = None,
    ) -> dict[str, Any]:
        filtered = [
            r.duration_ms for r in self._records
            if r.component == component
            and r.operation == operation
            and (cache_status is None or r.cache_status == cache_status)
        ]
        if not filtered:
            return {
                "avg_ms": 0.0,
                "p50_ms": 0.0,
                "p70_ms": 0.0,
                "p99_ms": 0.0,
                "p100_ms": 0.0,
                "count": 0,
            }
        values = sorted(filtered)
        avg = sum(values) / len(values)
        return {
            "avg_ms": round(avg, 2),
            "p50_ms": round(self.get_percentile_latency(component, operation, 50, cache_status), 2),
            "p70_ms": round(self.get_percentile_latency(component, operation, 70, cache_status), 2),
            "p99_ms": round(self.get_percentile_latency(component, operation, 99, cache_status), 2),
            "p100_ms": round(values[-1], 2),
            "count": len(values),
        }

    def summary(self) -> dict[str, Any]:
        components = {r.component for r in self._records}
        result: dict[str, Any] = {
            "counters": dict(self._counters),
            "cache_hit_rate": f"{self.cache_hit_rate:.1%}",
            "latency": {},
        }
        for comp in sorted(components):
            comp_records = [r for r in self._records if r.component == comp]
            operations = {r.operation for r in comp_records}
            result["latency"][comp] = {}
            for op in sorted(operations):
                result["latency"][comp][op] = {
                    **self.stage_summary(comp, op),
                    "cache_hit": self.stage_summary(comp, op, "hit"),
                    "cache_miss": self.stage_summary(comp, op, "miss"),
                }
        return result

    def reset(self) -> None:
        self._records.clear()
        self._counters.clear()


class Timer:
    """Context manager for timing operations."""

    def __init__(
        self,
        metrics: MetricsCollector,
        component: str,
        operation: str,
        cache_status: str | None = None,
    ) -> None:
        self._metrics = metrics
        self._component = component
        self._operation = operation
        self._cache_status = cache_status
        self._start: float = 0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._metrics.record_latency(
            self._component,
            self._operation,
            elapsed_ms,
            cache_status=self._cache_status,
        )
        self.elapsed_ms = elapsed_ms
