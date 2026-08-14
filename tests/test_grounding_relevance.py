import pytest
from voice_optimized_rag.core.guardrails import GuardrailEngine
from voice_optimized_rag.retrieval.vector_store import SearchResult


def test_guardrail_relevance_filter(config):
    config.guardrails_enabled = True
    config.guardrail_min_relevance_score = 0.25
    engine = GuardrailEngine(config)

    high_result = SearchResult(text="Manhattan project info", metadata={}, score=0.65, index=0)
    low_result = SearchResult(text="Unrelated sourdough info", metadata={}, score=0.11, index=1)

    filtered = engine.filter_relevant([high_result, low_result])
    assert len(filtered) == 1
    assert filtered[0].text == "Manhattan project info"


def test_guardrail_refuses_when_all_below_threshold(config):
    config.guardrails_enabled = True
    config.guardrail_min_relevance_score = 0.25
    config.guardrail_refusal_message = "I can't answer that from the available context."
    engine = GuardrailEngine(config)

    low_results = [
        SearchResult(text="Passage 1", metadata={}, score=0.24, index=0),
        SearchResult(text="Passage 2", metadata={}, score=0.18, index=1),
    ]

    decision = engine.evaluate("What are the things that you can do?", low_results, "miss")
    assert not decision.allowed
    assert decision.reason in ("low_relevance", "insufficient_context")
    assert decision.message == config.guardrail_refusal_message
    assert len(decision.relevant_results) == 0


def test_guardrail_partial_relevance_retains_only_strong_chunks(config):
    config.guardrails_enabled = True
    config.guardrail_min_relevance_score = 0.25
    engine = GuardrailEngine(config)

    results = [
        SearchResult(text="Relevant passage", metadata={}, score=0.55, index=0),
        SearchResult(text="Weak passage", metadata={}, score=0.15, index=1),
    ]

    decision = engine.evaluate("What is restorative justice?", results, "miss")
    assert decision.allowed
    assert len(decision.relevant_results) == 1
    assert decision.relevant_results[0].text == "Relevant passage"
