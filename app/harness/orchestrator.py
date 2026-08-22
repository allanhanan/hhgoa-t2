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
from app.harness.query_cache import get_cached_result, cache_result


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

    cached = get_cached_result(query_text)
    if cached is not None:
        cache_metrics = PipelineMetrics(
            total_ms=round((time.perf_counter() - pipeline_start) * 1000, 2)
        )
        return PipelineResult(
            answer=cached.answer,
            passages=cached.passages,
            metrics=cache_metrics,
            relevant=cached.relevant,
            relevance_score=cached.relevance_score,
            grounded=cached.grounded,
            answer_tier=cached.answer_tier,
            error=cached.error,
        )

    metrics = PipelineMetrics()

    # ── Stage 1: Input guardrail ──────────────────────────────────────
    with _timer(metrics, "guardrail_input_ms"):
        is_safe, reason = check_safety(query_text)
        if not is_safe:
            metrics.total_ms = (time.perf_counter() - pipeline_start) * 1000
            return PipelineResult(error=reason, metrics=metrics)

    # ── Stage 1.5: Query Language & Translation Routing ───────────────
    from app.preprocessing.query_language import translate_query_to_english
    with _timer(metrics, "translate_ms"):
        search_query, was_translated, orig_script = translate_query_to_english(query_text)

    # ── Stage 2: Embed ────────────────────────────────────────────────
    from app.embedder.encoder import encode, encode_and_binarize
    with _timer(metrics, "embed_ms"):
        if vector_db.is_ivf():
            float_emb = encode(search_query)
            binary_emb = None
        else:
            float_emb, binary_emb = encode_and_binarize(search_query)

    # ── Stage 3: Vector search ────────────────────────────────────────
    from app.config import TOP_K_BINARY, ANN_TOP_K
    with _timer(metrics, "search_ms"):
        q_vec = float_emb if vector_db.is_ivf() else binary_emb
        k_search = ANN_TOP_K if vector_db.is_ivf() else TOP_K_BINARY
        distances, ids = vector_db.search(q_vec, top_k=k_search)

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
        relevant, sim_score = is_relevant(float_emb, scored_passages=scored)
        if not relevant:
            metrics.total_ms = (time.perf_counter() - pipeline_start) * 1000
            return PipelineResult(
                answer="Your query doesn't seem related to the indexed content. "
                       "Please ask a question about the topics covered in the dataset.",
                passages=passages,
                metrics=metrics,
                relevant=False,
                relevance_score=sim_score,
                grounded=True,
                answer_tier="none",
            )

    # ── Stage 7: Answer extraction (3-tier cascade) ───────────────────
    passage_texts = [p.text for p in passages]
    answer = ""
    answer_tier = "verbatim_fallback"

    with _timer(metrics, "answer_ms"):
        # Tier 1: Heuristic fast path
        from app.answerer.heuristic import heuristic_extract
        heuristic_result = heuristic_extract(query_text, passage_texts[0] if passage_texts else "")

        from app.config import HEURISTIC_CONFIDENCE
        if heuristic_result.answer and heuristic_result.confidence >= HEURISTIC_CONFIDENCE:
            answer = heuristic_result.answer
            answer_tier = "heuristic"

        else:
            # Tier 2: Extractive QA model (ONNX)
            try:
                from app.answerer.extractive_qa import answer as qa_answer
                qa_result = qa_answer(query_text, passage_texts)
                if qa_result.text and qa_result.confidence > 0.0:
                    answer = qa_result.text
                    answer_tier = "qa_model"
            except Exception:
                pass

        # Tier 3: Passage verbatim fallback
        if not answer:
            answer_tier = "verbatim_fallback"
            if passages:
                answer = passages[0].text
            else:
                answer = "No relevant context found in dataset."

    metrics.answer_extract_ms = metrics.answer_ms

    # ── Stage 8: Output guardrail ─────────────────────────────────────
    with _timer(metrics, "guardrail_output_ms"):
        if answer_tier == "qa_model":
            grounded, overlap = check_grounding(answer, passage_texts)
        else:
            grounded = True

    metrics.total_ms = (time.perf_counter() - pipeline_start) * 1000

    result = PipelineResult(
        answer=answer,
        passages=passages,
        metrics=metrics,
        relevant=relevant,
        relevance_score=sim_score,
        grounded=grounded,
        answer_tier=answer_tier,
    )
    cache_result(query_text, result)
    return result


async def warmup_pipeline() -> None:
    """Run full-pipeline dummy queries across all stages and branches before timing/serving."""
    from app.harness.query_cache import clear_cache
    dummy_queries = [
        "What is retrieval augmented generation?",
        "Who invented the telephone?",
        "இந்தியாவின் தலைநகரம் எது",
        "What is artificial intelligence?",
    ]
    for q in dummy_queries:
        try:
            await run_pipeline(q)
        except Exception:
            pass
    clear_cache()

