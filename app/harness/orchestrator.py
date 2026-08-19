"""Pipeline orchestrator — full RAG pipeline with per-stage timing.

Uses a 3-tier answer cascade:
  Tier 1: Heuristic fast-path (regex + question-type, <1ms)
  Tier 2: Extractive QA model (ONNX MiniLM span extraction, ~30ms)
  Tier 3: Passage verbatim fallback (0ms)

No generative SLM. No autoregressive decoding. Zero hallucination.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from app.models import PipelineMetrics, PipelineResult, PassageResult
from app.embedder.encoder import encode_and_binarize
from app.retriever import vector_db, payload_store
from app.retriever.rescorer import rescore
from app.guardrails.safety import check_safety
from app.guardrails.relevance import is_relevant
from app.guardrails.grounding import check_grounding


@contextmanager
def _timer(metrics: PipelineMetrics, field: str):
    """Context manager that records elapsed time into a PipelineMetrics field."""
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    setattr(metrics, field, elapsed_ms)


async def run_pipeline(query_text: str, top_k: int = 5) -> PipelineResult:
    """Execute the full RAG pipeline: guardrail → embed → search → rescore → fetch → answer.

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
    from app.config import TOP_K_BINARY
    with _timer(metrics, "search_ms"):
        distances, ids = vector_db.search(binary_emb, top_k=TOP_K_BINARY)

    # ── Stage 4: Rescore ──────────────────────────────────────────────
    with _timer(metrics, "rescore_ms"):
        scored = rescore(float_emb, ids, top_k=top_k)

    # ── Stage 5: Payload fetch ────────────────────────────────────────
    with _timer(metrics, "payload_ms"):
        scored_ids = [s[0] for s in scored]
        scores_map = {s[0]: s[1] for s in scored}
        passages = payload_store.fetch(scored_ids)
        for p in passages:
            p.score = scores_map.get(p.id, 0.0)

    # ── Stage 6: Context guardrail ────────────────────────────────────
    with _timer(metrics, "guardrail_context_ms"):
        top_score = passages[0].score if passages else None
        relevant, sim_score = is_relevant(float_emb, max_passage_score=top_score)
        if not relevant:
            metrics.total_ms = (time.perf_counter() - pipeline_start) * 1000
            return PipelineResult(
                answer="Your query doesn't seem related to the indexed content. "
                       "Please ask a question about the topics covered in the dataset.",
                passages=passages,
                metrics=metrics,
            )

    # ── Stage 7: Answer extraction (3-tier cascade) ───────────────────
    passage_texts = [p.text for p in passages]
    answer = ""

    with _timer(metrics, "answer_ms"):
        # Tier 1: Heuristic fast path
        from app.answerer.heuristic import heuristic_extract
        heuristic_result = heuristic_extract(query_text, passage_texts[0] if passage_texts else "")

        if heuristic_result.answer and heuristic_result.confidence >= 0.7:
            answer = heuristic_result.answer
        else:
            # Tier 2: Extractive QA model (ONNX)
            try:
                from app.answerer.extractive_qa import answer as qa_answer
                qa_result = qa_answer(query_text, passage_texts)
                if qa_result.text and qa_result.confidence > 0.0:
                    answer = qa_result.text
            except Exception:
                pass

        # Tier 3: Passage verbatim fallback
        if not answer:
            if passages:
                answer = passages[0].text
            else:
                answer = "No relevant context found in dataset."

    # Keep generate_ttft_ms populated for backward compatibility
    metrics.generate_ttft_ms = metrics.answer_ms

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
