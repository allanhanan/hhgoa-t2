"""Generator module exposing generate_answer for eval loop and pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class GeneratedAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str


def generate_answer(query: str, results: list[Any]) -> GeneratedAnswer:
    """Generate or extract an answer given a query and retrieved context results.

    Args:
        query: The user or evaluation question.
        results: List of objects having .text and .source attributes.

    Returns:
        GeneratedAnswer object with text, grounded, generation_ms, and model.
    """
    t0 = time.perf_counter()

    if not results:
        return GeneratedAnswer(
            text="The provided documents do not contain information to answer this question.",
            grounded=False,
            generation_ms=(time.perf_counter() - t0) * 1000,
            model="extractive-cascade",
        )

    passage_texts = [r.text for r in results if hasattr(r, "text") and r.text and r.text.strip()]
    if not passage_texts:
        return GeneratedAnswer(
            text="The provided documents do not contain information to answer this question.",
            grounded=False,
            generation_ms=(time.perf_counter() - t0) * 1000,
            model="extractive-cascade",
        )

    # 1. Tier 1: Heuristic fast-path
    try:
        from app.answerer.heuristic import heuristic_extract
        from app.config import HEURISTIC_CONFIDENCE

        heuristic_res = heuristic_extract(query, passage_texts[0])
        if heuristic_res.answer and heuristic_res.confidence >= HEURISTIC_CONFIDENCE:
            return GeneratedAnswer(
                text=heuristic_res.answer,
                grounded=True,
                generation_ms=(time.perf_counter() - t0) * 1000,
                model="heuristic",
            )
    except Exception:
        pass

    # 2. Tier 2: Extractive QA model (ONNX MiniLM)
    try:
        from app.answerer.extractive_qa import answer as qa_answer

        qa_res = qa_answer(query, passage_texts)
        if qa_res.text and qa_res.confidence > 0.0:
            ans_clean = qa_res.text.strip().lower().rstrip(".?! ")
            q_clean = query.strip().lower().rstrip(".?! ")

            # Avoid false confidence if extracted span is merely an echo of the query
            if ans_clean != q_clean and not (len(ans_clean) > 8 and ans_clean in q_clean):
                from app.guardrails.grounding import check_grounding

                is_grounded, _ = check_grounding(qa_res.text, passage_texts)
                if is_grounded:
                    return GeneratedAnswer(
                        text=qa_res.text,
                        grounded=True,
                        generation_ms=(time.perf_counter() - t0) * 1000,
                        model="minilm-qa",
                    )
    except Exception:
        pass

    # No confident answer found in context passages -> ungrounded / abstention
    return GeneratedAnswer(
        text="The provided documents do not contain sufficient information to answer this question.",
        grounded=False,
        generation_ms=(time.perf_counter() - t0) * 1000,
        model="extractive-cascade",
    )
