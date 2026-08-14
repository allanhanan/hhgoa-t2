"""Integration tests: multi-turn conversation showing prefetch benefit."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.memory_router import MemoryRouter



@pytest.mark.asyncio
async def test_multi_turn_conversation(config, mock_llm, mock_embeddings):
    """Test a multi-turn conversation demonstrating the dual-agent pattern.

    The slow thinker should prefetch context after the first query,
    making subsequent related queries faster via cache hits.
    """
    config.slow_thinker_rate_limit = 0  # No rate limiting for tests
    config.cache_similarity_threshold = 0.3  # Lower threshold for mock embeddings

    router = MemoryRouter(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
    )
    await router.start(log_level="WARNING")

    # Ingest some related documents
    docs = [
        "Our basic plan costs $10 per month with 100 API calls.",
        "The pro plan is $50 per month with 10000 API calls.",
        "Enterprise pricing is custom, contact sales for a quote.",
        "All plans include email support. Pro and Enterprise get phone support.",
        "Annual billing saves 20% compared to monthly billing.",
    ]
    await router.ingest_texts(docs)

    # Turn 1: First query (likely cache miss — cold start)
    response1 = await router.query("What pricing plans do you offer?")
    assert response1

    # Give the slow thinker time to predict and prefetch
    await asyncio.sleep(2.0)

    # Turn 2: Related follow-up (should benefit from prefetching)
    response2 = await router.query("What about enterprise pricing?")
    assert response2

    # Turn 3: Another related follow-up
    response3 = await router.query("Do you offer annual billing discounts?")
    assert response3

    # Verify the system processed events
    history = router.stream.history
    assert len(history) >= 6  # 3 queries + 3 responses

    # Verify metrics show activity (prefetch or cache lookups)
    summary = router.metrics.summary()
    total_counters = sum(summary["counters"].values())
    assert total_counters > 0  # System was actively processing

    await router.stop()


@pytest.mark.asyncio
async def test_topic_shift(config, mock_llm, mock_embeddings):
    """Test that the system handles topic shifts gracefully."""
    config.slow_thinker_rate_limit = 0

    router = MemoryRouter(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
    )
    await router.start(log_level="WARNING")

    await router.ingest_texts([
        "Python SDK supports async operations.",
        "JavaScript SDK uses Promise-based API.",
        "Pricing starts at $10 per month.",
    ])

    # Topic 1: Technical
    await router.query("Tell me about the Python SDK")
    await asyncio.sleep(0.5)

    # Topic 2: Pricing (shift)
    await router.query("What about pricing?")

    # Both should get responses
    history = router.stream.history
    assert len(history) >= 4

    await router.stop()


@pytest.mark.asyncio
async def test_empty_knowledge_base(config, mock_llm, mock_embeddings):
    """Test system works even with no documents ingested."""
    router = MemoryRouter(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
    )
    await router.start(log_level="WARNING")

    # Query with empty knowledge base
    response = await router.query("What is the meaning of life?")
    assert response  # Should still respond from parametric knowledge

    await router.stop()
