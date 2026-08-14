"""Lightweight guardrails for grounded RAG answers."""

from __future__ import annotations

from dataclasses import dataclass, field

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.retrieval.vector_store import SearchResult


@dataclass(frozen=True)
class GuardrailDecision:
    """Decision returned before an LLM call.

    When allowed, ``relevant_results`` holds the retrieval results that passed
    the relevance floor. These are the ONLY chunks that should be sent to the
    LLM — weak/unrelated passages must not influence generation.
    """

    allowed: bool
    reason: str = ""
    message: str = ""
    relevant_results: list[SearchResult] = field(default_factory=list)


class GuardrailEngine:
    """Rule-based checks that keep Fast Talker grounded and cheap.

    All checks are O(1) / O(k) CPU — no LLM calls, so guardrails add <1ms.
    """

    _unsafe_terms = {
        "make a bomb",
        "bypass authentication",
        "steal password",
        "malware",
        "phishing",
    }

    def __init__(self, config: VORConfig) -> None:
        self._config = config

    def filter_relevant(self, results: list[SearchResult]) -> list[SearchResult]:
        """Keep only results at or above the configured relevance floor.

        A floor of ``0.0`` (the default) disables filtering — every result is kept,
        preserving the historical "0 = no relevance floor" semantics.
        """
        min_score = self._config.guardrail_min_relevance_score
        if min_score <= 0:
            return list(results)
        return [r for r in results if r.score >= min_score]

    def evaluate(
        self,
        query: str,
        results: list[SearchResult],
        cache_status: str,
    ) -> GuardrailDecision:
        """Pre-generation guardrail — the relevance/grounding gate.

        Checks (in order, short-circuit on first failure):
          1. Guardrails disabled → always allow.
          2. Unsafe content in query → deny.
          3. Off-topic (if off_topic_terms configured) → deny.
          4. Drop results below the configured relevance floor.
          5. No results remain after the floor → deny (low_relevance).
          6. Fewer than guardrail_min_context_chunks remain → deny (insufficient_context).
          7. Best chunk still below the floor → deny (defensive; 4 already removed them).

        On success, ``relevant_results`` contains ONLY the chunks that met the
        relevance floor. The caller must send those — and nothing else — to the LLM.
        """
        if not self._config.guardrails_enabled:
            return GuardrailDecision(allowed=True)

        lowered = query.lower()

        # Hard-block unsafe requests
        if any(term in lowered for term in self._unsafe_terms):
            return self._deny("unsafe_query", "I can't help with that request.")

        # Optional topic gating: only enforce when off_topic_terms is non-empty
        off_topic_terms = [t.lower() for t in self._config.guardrail_off_topic_terms]
        if off_topic_terms and not any(term in lowered for term in off_topic_terms):
            return self._deny("off_topic", self._config.guardrail_refusal_message)

        # Relevance gate: keep only chunks at/above the configured floor.
        # Retrieval ≠ relevance — Qdrant/FAISS return nearest vectors even when the
        # query is unrelated to the index, so the floor decides what is answerable.
        relevant = self.filter_relevant(results)

        if not relevant:
            reason = "low_relevance" if results else "insufficient_context"
            return self._deny(reason, self._config.guardrail_refusal_message)

        # Require at least N relevant context chunks (default 1).
        if len(relevant) < self._config.guardrail_min_context_chunks:
            return self._deny("insufficient_context", self._config.guardrail_refusal_message)

        # Defensive top-score check (results were already filtered by the floor).
        min_score = self._config.guardrail_min_relevance_score
        if min_score > 0:
            best_score = max(r.score for r in relevant)
            if best_score < min_score:
                return self._deny("low_relevance", self._config.guardrail_refusal_message)

        return GuardrailDecision(allowed=True, reason=cache_status, relevant_results=relevant)

    def validate_answer(self, answer: str, context: str) -> GuardrailDecision:
        """Post-generation guardrail: reject empty answers."""
        if not self._config.guardrails_enabled:
            return GuardrailDecision(allowed=True)
        if not answer.strip():
            return self._deny("empty_answer", self._config.guardrail_refusal_message)
        return GuardrailDecision(allowed=True)

    def _deny(self, reason: str, message: str) -> GuardrailDecision:
        return GuardrailDecision(allowed=False, reason=reason, message=message)
