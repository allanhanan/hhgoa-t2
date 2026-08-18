"""Hard timeout enforcement for the pipeline."""
from __future__ import annotations

import asyncio

from app.config import LATENCY_BUDGET_MS
from app.models import PipelineResult


async def with_timeout(coro, timeout_ms: float | None = None) -> PipelineResult:
    """Run a coroutine with a hard timeout.

    If the timeout is exceeded, returns a partial result with an error message.
    """
    timeout_s = (timeout_ms or LATENCY_BUDGET_MS) / 1000.0

    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError:
        return PipelineResult(
            answer="Processing took too long. Please try a shorter query.",
            error=f"Pipeline exceeded {timeout_ms or LATENCY_BUDGET_MS}ms timeout",
        )
