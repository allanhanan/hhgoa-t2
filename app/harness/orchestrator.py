"""Pipeline orchestrator — full RAG pipeline with per-stage timing."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import AsyncIterator

from app.models import PipelineMetrics, PipelineResult, PassageResult
from app.embedder.encoder import encode_and_binarize
from app.retriever import vector_db, payload_store
from app.retriever.rescorer import rescore
from app.guardrails.safety import check_safety
from app.guardrails.relevance import is_relevant
from app.guardrails.grounding import check_grounding
from app.harness.circuit_breaker import CircuitBreaker

# Circuit breakers for LLM providers
_local_llm_cb = CircuitBreaker("local_llm", failure_threshold=5, recovery_timeout=30)
_groq_cb = CircuitBreaker("groq", failure_threshold=5, recovery_timeout=30)


@contextmanager
def _timer(metrics: PipelineMetrics, field: str):
    """Context manager that records elapsed time into a PipelineMetrics field."""
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    setattr(metrics, field, elapsed_ms)


async def run_pipeline(query_text: str, top_k: int = 5) -> PipelineResult:
    """Execute the full RAG pipeline: guardrail → embed → search → rescore → fetch → generate.

    Returns a PipelineResult with the answer, passages, and per-stage metrics.
    """
    pipeline_start = time.perf_counter()
    metrics = PipelineMetrics()

    # ── Stage 1: Input guardrail ──────────────────────────────────────
    with _timer(metrics, "guardrail_input_ms"):
        is_safe, reason = check_safety(query_text)
        if not is_safe:
            metrics.total_ms = (time.perf_counter() - pipeline_start) * 1000
            return PipelineResult(error=reason, metrics=metrics)

    # ── Stage 2: Embed ────────────────────────────────────────────────
    with _timer(metrics, "embed_ms"):
        float_emb, binary_emb = encode_and_binarize(query_text)

    # ── Stage 3: Binary search ────────────────────────────────────────
    with _timer(metrics, "search_ms"):
        distances, ids = vector_db.search(binary_emb, top_k=100)

    # ── Stage 4: Rescore ──────────────────────────────────────────────
    with _timer(metrics, "rescore_ms"):
        scored = rescore(float_emb, ids, top_k=top_k)

    # ── Stage 5: Payload fetch ────────────────────────────────────────
    with _timer(metrics, "payload_ms"):
        scored_ids = [s[0] for s in scored]
        scores_map = {s[0]: s[1] for s in scored}
        passages = payload_store.fetch(scored_ids)
        # Attach scores
        for p in passages:
            p.score = scores_map.get(p.id, 0.0)

    # ── Stage 6: Context guardrail ────────────────────────────────────
    with _timer(metrics, "guardrail_context_ms"):
        relevant, sim_score = is_relevant(float_emb)
        if not relevant:
            metrics.total_ms = (time.perf_counter() - pipeline_start) * 1000
            return PipelineResult(
                answer="Your query doesn't seem related to the indexed content. Please ask a question about the topics covered in the dataset.",
                passages=passages,
                metrics=metrics,
            )

    # ── Stage 7: Generate ─────────────────────────────────────────────
    passage_texts = [p.text for p in passages]
    answer = ""
    ttft_recorded = False

    try:
        if _local_llm_cb.is_available():
            from app.generator.local_llm import generate_stream
            gen_start = time.perf_counter()
            async for token in generate_stream(query_text, passage_texts):
                if not ttft_recorded:
                    metrics.generate_ttft_ms = (time.perf_counter() - gen_start) * 1000
                    ttft_recorded = True
                answer += token
            metrics.generate_total_ms = (time.perf_counter() - gen_start) * 1000
            _local_llm_cb.record_success()
        else:
            raise ConnectionError("Local LLM circuit breaker open")
    except Exception as e:
        _local_llm_cb.record_failure()
        # Try Groq fallback
        try:
            if _groq_cb.is_available():
                from app.generator.groq_fallback import generate_stream as groq_stream
                gen_start = time.perf_counter()
                async for token in groq_stream(query_text, passage_texts):
                    if not ttft_recorded:
                        metrics.generate_ttft_ms = (time.perf_counter() - gen_start) * 1000
                        ttft_recorded = True
                    answer += token
                metrics.generate_total_ms = (time.perf_counter() - gen_start) * 1000
                _groq_cb.record_success()
            else:
                answer = f"LLM unavailable (local: {e}). Retrieved passages are shown below."
        except Exception as groq_e:
            _groq_cb.record_failure()
            answer = f"Both LLM providers failed. Local: {e}. Groq: {groq_e}"

    # ── Stage 8: Output guardrail ─────────────────────────────────────
    with _timer(metrics, "guardrail_output_ms"):
        grounded, overlap = check_grounding(answer, passage_texts)

    metrics.total_ms = (time.perf_counter() - pipeline_start) * 1000

    return PipelineResult(
        answer=answer,
        passages=passages,
        metrics=metrics,
        grounded=grounded,
    )
