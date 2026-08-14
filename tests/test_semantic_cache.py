"""Tests for the semantic cache."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from voice_optimized_rag.core.semantic_cache import SemanticCache
from voice_optimized_rag.utils.metrics import MetricsCollector


def _random_embedding(dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


@pytest.mark.asyncio
async def test_put_and_get(cache: SemanticCache, dim: int):
    """Test basic put and get operations."""
    emb = _random_embedding(dim, seed=1)
    await cache.put(emb, "Hello world", metadata={"source": "test"})

    assert cache.size == 1

    # Exact same embedding should be a hit
    results = await cache.get(emb)
    assert len(results) == 1
    assert results[0].text == "Hello world"
    assert results[0].metadata == {"source": "test"}


@pytest.mark.asyncio
async def test_similar_embedding_hit(cache: SemanticCache, dim: int):
    """Test that similar embeddings produce cache hits."""
    emb = _random_embedding(dim, seed=1)
    await cache.put(emb, "Test content")

    # Slightly perturbed embedding should still hit
    noise = np.random.RandomState(99).randn(dim).astype(np.float32) * 0.05
    similar = emb + noise
    similar /= np.linalg.norm(similar)

    results = await cache.get(similar)
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_dissimilar_embedding_miss(cache: SemanticCache, dim: int):
    """Test that dissimilar embeddings produce cache misses."""
    emb1 = _random_embedding(dim, seed=1)
    emb2 = _random_embedding(dim, seed=999)  # Very different seed

    await cache.put(emb1, "Content A")

    results = await cache.get(emb2, similarity_threshold=0.95)
    # With high threshold, very different embeddings should miss
    assert len(results) == 0


@pytest.mark.asyncio
async def test_ttl_expiration(dim: int, metrics: MetricsCollector):
    """Test that entries expire after TTL."""
    cache = SemanticCache(
        dimension=dim,
        default_ttl=0.1,  # 100ms TTL
        similarity_threshold=0.3,
        metrics=metrics,
    )

    emb = _random_embedding(dim, seed=1)
    await cache.put(emb, "Temporary content", ttl=0.1)

    # Immediate get should work
    results = await cache.get(emb)
    assert len(results) == 1

    # Wait for expiration
    await asyncio.sleep(0.15)

    results = await cache.get(emb)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_max_size_eviction(dim: int, metrics: MetricsCollector):
    """Test LRU eviction when cache exceeds max size."""
    cache = SemanticCache(
        dimension=dim,
        max_size=3,
        similarity_threshold=0.3,
        metrics=metrics,
    )

    for i in range(5):
        emb = _random_embedding(dim, seed=i)
        await cache.put(emb, f"Content {i}")

    # Cache should not exceed max size
    assert cache.size <= 3


@pytest.mark.asyncio
async def test_clear(cache: SemanticCache, dim: int):
    """Test clearing the cache."""
    for i in range(5):
        emb = _random_embedding(dim, seed=i)
        await cache.put(emb, f"Content {i}")

    assert cache.size == 5
    await cache.clear()
    assert cache.size == 0


@pytest.mark.asyncio
async def test_dedup_near_duplicates(cache: SemanticCache, dim: int):
    """Test that near-duplicate entries are updated instead of added."""
    emb = _random_embedding(dim, seed=1)
    await cache.put(emb, "Version 1")
    await cache.put(emb, "Version 2")

    # Should still be 1 entry (updated, not duplicated)
    assert cache.size == 1

    results = await cache.get(emb)
    assert results[0].text == "Version 2"


@pytest.mark.asyncio
async def test_metrics_tracking(dim: int, metrics: MetricsCollector):
    """Test that cache tracks hit/miss metrics."""
    cache = SemanticCache(
        dimension=dim,
        similarity_threshold=0.3,
        metrics=metrics,
    )

    emb = _random_embedding(dim, seed=1)
    await cache.put(emb, "Content")

    # Hit
    await cache.get(emb)
    assert metrics.get_counter("cache_hit") == 1

    # Miss (very different embedding)
    miss_emb = _random_embedding(dim, seed=999)
    await cache.get(miss_emb, similarity_threshold=0.99)
    assert metrics.get_counter("cache_miss") >= 1

    assert metrics.cache_hit_rate > 0


@pytest.mark.asyncio
async def test_multiple_results(cache: SemanticCache, dim: int):
    """Test retrieving multiple cached entries."""
    base = _random_embedding(dim, seed=1)

    # Add several related entries with slight variations
    for i in range(5):
        noise = np.random.RandomState(i + 100).randn(dim).astype(np.float32) * 0.1
        emb = base + noise
        emb /= np.linalg.norm(emb)
        await cache.put(emb, f"Related content {i}")

    results = await cache.get(base, top_k=3, similarity_threshold=0.3)
    assert len(results) >= 1
