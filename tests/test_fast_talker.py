"""Tests for the Fast Talker foreground agent."""

from __future__ import annotations

import numpy as np
import pytest

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.conversation_stream import ConversationStream
from voice_optimized_rag.core.fast_talker import FastTalker
from voice_optimized_rag.core.semantic_cache import SemanticCache
from voice_optimized_rag.retrieval.vector_store import FAISSVectorStore
from voice_optimized_rag.utils.metrics import MetricsCollector



@pytest.mark.asyncio
async def test_fast_talker_with_cache_hit(
    config, mock_llm, mock_embeddings, vector_store, cache, stream, metrics, dim
):
    """Test response generation with a cache hit."""
    # Pre-populate cache
    emb = await mock_embeddings.embed_single("What is pricing?")
    await cache.put(emb, "Pricing starts at $10/month.", relevance_score=0.9)

    talker = FastTalker(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
        vector_store=vector_store, cache=cache, stream=stream, metrics=metrics,
    )

    response = await talker.respond("What is pricing?")

    assert response  # Got a response
    assert mock_llm.call_count == 1
    # LLM should have been called with context from cache
    assert mock_llm.last_context != ""  # Context was provided


@pytest.mark.asyncio
async def test_fast_talker_cache_miss_fallback(
    config, mock_llm, mock_embeddings, vector_store, cache, stream, metrics, dim
):
    """Test fallback to vector store on cache miss."""
    # Add documents to vector store (not cache)
    texts = ["Pricing info: $10/month basic plan."]
    rng = np.random.RandomState(42)
    embeddings = rng.randn(1, dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    vector_store.add_documents(texts, embeddings)

    config.fast_talker_fallback_to_retrieval = True

    talker = FastTalker(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
        vector_store=vector_store, cache=cache, stream=stream, metrics=metrics,
    )

    response = await talker.respond("What is the pricing?")
    assert response


@pytest.mark.asyncio
async def test_fast_talker_no_context(
    config, mock_llm, mock_embeddings, vector_store, cache, stream, metrics
):
    """Test response with no context available (empty cache and store)."""
    config.fast_talker_fallback_to_retrieval = True

    talker = FastTalker(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
        vector_store=vector_store, cache=cache, stream=stream, metrics=metrics,
    )

    response = await talker.respond("Random question with no docs")
    assert response  # Should still respond (from parametric knowledge)
    assert mock_llm.last_context == ""  # No context was available


@pytest.mark.asyncio
async def test_fast_talker_streaming(
    config, mock_llm, mock_embeddings, vector_store, cache, stream, metrics
):
    """Test streaming response."""
    talker = FastTalker(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
        vector_store=vector_store, cache=cache, stream=stream, metrics=metrics,
    )

    chunks = []
    async for chunk in talker.respond_stream("Test query"):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert full_response.strip()  # Non-empty


@pytest.mark.asyncio
async def test_fast_talker_latency_tracking(
    config, mock_llm, mock_embeddings, vector_store, cache, stream, metrics
):
    """Test that latency metrics are recorded."""
    talker = FastTalker(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
        vector_store=vector_store, cache=cache, stream=stream, metrics=metrics,
    )

    await talker.respond("Test query")

    avg = metrics.get_avg_latency("fast_talker", "total_response")
    assert avg > 0  # Latency was recorded
