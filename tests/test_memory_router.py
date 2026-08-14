"""Tests for the Memory Router orchestrator."""

from __future__ import annotations

import numpy as np
import pytest

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.memory_router import MemoryRouter



@pytest.mark.asyncio
async def test_router_lifecycle(config, mock_llm, mock_embeddings):
    """Test router start and stop."""
    router = MemoryRouter(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
    )

    await router.start(log_level="WARNING")
    assert router._running

    await router.stop()
    assert not router._running


@pytest.mark.asyncio
async def test_router_ingest_texts(config, mock_llm, mock_embeddings):
    """Test ingesting raw texts."""
    router = MemoryRouter(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
    )

    texts = ["Document one.", "Document two.", "Document three."]
    count = await router.ingest_texts(texts)

    assert count == 3


@pytest.mark.asyncio
async def test_router_query(config, mock_llm, mock_embeddings):
    """Test basic query flow."""
    router = MemoryRouter(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
    )
    await router.start(log_level="WARNING")

    # Ingest some docs
    await router.ingest_texts(["Pricing is $10/month.", "We support Python and JS."])

    response = await router.query("What is the pricing?")
    assert response
    assert mock_llm.call_count >= 1

    await router.stop()


@pytest.mark.asyncio
async def test_router_query_stream(config, mock_llm, mock_embeddings):
    """Test streaming query flow."""
    router = MemoryRouter(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
    )
    await router.start(log_level="WARNING")

    await router.ingest_texts(["Test document content."])

    chunks = []
    async for chunk in router.query_stream("Tell me about the test"):
        chunks.append(chunk)

    assert len(chunks) > 0
    await router.stop()


@pytest.mark.asyncio
async def test_router_conversation_history(config, mock_llm, mock_embeddings):
    """Test that conversation history is maintained."""
    router = MemoryRouter(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
    )
    await router.start(log_level="WARNING")

    await router.query("First question")
    await router.query("Second question")

    history = router.stream.history
    # Should have at least user utterance + agent response for each query
    assert len(history) >= 4

    await router.stop()


@pytest.mark.asyncio
async def test_router_metrics(config, mock_llm, mock_embeddings):
    """Test that metrics are collected."""
    router = MemoryRouter(
        config=config, llm=mock_llm, embedding_provider=mock_embeddings,
    )
    await router.start(log_level="WARNING")

    await router.ingest_texts(["Test content."])
    await router.query("Test query")

    summary = router.metrics.summary()
    assert "counters" in summary
    assert "latency" in summary

    await router.stop()
