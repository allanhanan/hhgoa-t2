"""Tests for the Slow Thinker background agent."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.conversation_stream import (
    ConversationStream,
    EventType,
    StreamEvent,
)
from voice_optimized_rag.core.semantic_cache import SemanticCache
from voice_optimized_rag.core.slow_thinker import SlowThinker
from voice_optimized_rag.utils.metrics import MetricsCollector


def _populate_vector_store(vector_store, embedding_provider, dim):
    """Add sample documents to the vector store."""
    texts = [
        "Our pricing starts at $10/month for the basic plan.",
        "The enterprise plan includes dedicated support and SLA.",
        "API rate limits are 1000 requests per minute.",
        "We support Python, JavaScript, and Go SDKs.",
        "Authentication uses OAuth 2.0 with JWT tokens.",
    ]
    rng = np.random.RandomState(42)
    embeddings = rng.randn(len(texts), dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    vector_store.add_documents(
        texts, embeddings,
        [{"source": f"doc{i}.txt"} for i in range(len(texts))],
    )


@pytest.mark.asyncio
async def test_slow_thinker_starts_and_stops(
    config, mock_llm, mock_embeddings, vector_store, cache, stream, metrics
):
    """Test that the slow thinker can start and stop cleanly."""
    thinker = SlowThinker(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
        vector_store=vector_store, cache=cache, stream=stream, metrics=metrics,
    )

    await thinker.start()
    assert thinker._running
    await asyncio.sleep(0.1)
    await thinker.stop()
    assert not thinker._running


@pytest.mark.asyncio
async def test_slow_thinker_processes_utterance(
    config, mock_llm, mock_embeddings, vector_store, cache, stream, metrics, dim
):
    """Test that the slow thinker processes user utterances and caches results."""
    _populate_vector_store(vector_store, mock_embeddings, dim)

    config.slow_thinker_rate_limit = 0  # No rate limiting for tests

    thinker = SlowThinker(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
        vector_store=vector_store, cache=cache, stream=stream, metrics=metrics,
    )

    await thinker.start()
    # Let the background task subscribe before publishing
    await asyncio.sleep(0.1)

    # Publish a user utterance
    await stream.publish(StreamEvent(
        event_type=EventType.USER_UTTERANCE,
        text="Tell me about your pricing",
    ))

    # Give the slow thinker time to process
    await asyncio.sleep(2.0)

    await thinker.stop()

    # Check that predictions were made
    assert mock_llm.call_count > 0
    # Check that prefetch operations occurred
    assert metrics.get_counter("prefetch_operations") > 0


@pytest.mark.asyncio
async def test_slow_thinker_handles_priority_retrieval(
    config, mock_llm, mock_embeddings, vector_store, cache, stream, metrics, dim
):
    """Test that priority retrieval events are handled."""
    _populate_vector_store(vector_store, mock_embeddings, dim)

    thinker = SlowThinker(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
        vector_store=vector_store, cache=cache, stream=stream, metrics=metrics,
    )

    await thinker.start()
    await asyncio.sleep(0.1)

    await stream.publish(StreamEvent(
        event_type=EventType.PRIORITY_RETRIEVAL,
        text="What are the rate limits?",
    ))

    await asyncio.sleep(2.0)
    await thinker.stop()

    assert metrics.get_counter("prefetch_operations") > 0


@pytest.mark.asyncio
async def test_slow_thinker_rate_limiting(
    config, mock_llm, mock_embeddings, vector_store, cache, stream, metrics, dim
):
    """Test that the slow thinker respects rate limiting."""
    _populate_vector_store(vector_store, mock_embeddings, dim)

    config.slow_thinker_rate_limit = 10.0  # High rate limit

    thinker = SlowThinker(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
        vector_store=vector_store, cache=cache, stream=stream, metrics=metrics,
    )

    await thinker.start()

    # Send multiple utterances rapidly
    for i in range(3):
        await stream.publish(StreamEvent(
            event_type=EventType.USER_UTTERANCE,
            text=f"Question {i}",
        ))

    await asyncio.sleep(0.5)
    await thinker.stop()

    # With high rate limit, only the first should be processed
    prefetch_count = metrics.get_counter("prefetch_operations")
    assert prefetch_count <= 2  # At most 1 for the query + 1 for predictions
