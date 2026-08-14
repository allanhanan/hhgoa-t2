"""Semantic similarity cache for zero-latency context retrieval.

This is the critical bridge between the Slow Thinker and Fast Talker.
The Slow Thinker writes pre-fetched context here; the Fast Talker reads from it.
Uses a small FAISS index internally for sub-millisecond semantic similarity lookup.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import faiss
import numpy as np

from voice_optimized_rag.utils.logging import get_logger
from voice_optimized_rag.utils.metrics import MetricsCollector

logger = get_logger("semantic_cache")


@dataclass
class CachedContext:
    """A cached context entry."""
    text: str
    metadata: dict
    embedding: np.ndarray
    relevance_score: float
    created_at: float = field(default_factory=time.time)
    ttl: float = 300.0  # seconds
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl


class SemanticCache:
    """In-memory semantic similarity cache with FAISS-backed lookup.

    Provides sub-millisecond retrieval of pre-fetched context by performing
    cosine similarity search over cached embeddings.
    """

    def __init__(
        self,
        dimension: int,
        max_size: int = 1000,
        default_ttl: float = 300.0,
        similarity_threshold: float = 0.75,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._dimension = dimension
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._similarity_threshold = similarity_threshold
        self._metrics = metrics or MetricsCollector()
        self._lock = asyncio.Lock()

        # Cache storage
        self._entries: list[CachedContext] = []
        self._index = faiss.IndexFlatIP(dimension)  # Inner product for cosine sim

    @property
    def size(self) -> int:
        return len(self._entries)

    async def put(
        self,
        query_embedding: np.ndarray,
        text: str,
        metadata: dict | None = None,
        relevance_score: float = 1.0,
        ttl: float | None = None,
    ) -> None:
        """Add a context entry to the cache.

        Args:
            query_embedding: The embedding vector this context is relevant to.
            text: The context text.
            metadata: Optional metadata.
            relevance_score: How relevant this context is (0-1).
            ttl: Time-to-live in seconds (None = use default).
        """
        async with self._lock:
            # Check if we already have very similar content
            if self._index.ntotal > 0:
                query = query_embedding.reshape(1, -1).astype(np.float32).copy()
                faiss.normalize_L2(query)
                scores, indices = self._index.search(query, 1)
                if scores[0][0] > 0.95 and indices[0][0] != -1:
                    # Near-duplicate — update existing entry instead
                    idx = int(indices[0][0])
                    if idx < len(self._entries):
                        self._entries[idx].text = text
                        self._entries[idx].relevance_score = relevance_score
                        self._entries[idx].created_at = time.time()
                        return

            # Evict expired entries
            self._evict_expired()

            # Evict LRU if at capacity
            if len(self._entries) >= self._max_size:
                self._evict_lru()

            # Add new entry
            embedding = query_embedding.reshape(1, -1).astype(np.float32).copy()
            faiss.normalize_L2(embedding)
            self._index.add(embedding)
            self._entries.append(CachedContext(
                text=text,
                metadata=metadata or {},
                embedding=query_embedding,
                relevance_score=relevance_score,
                ttl=ttl or self._default_ttl,
            ))

    async def get(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float | None = None,
    ) -> list[CachedContext]:
        """Retrieve relevant cached context.

        Args:
            query_embedding: The query embedding to search for.
            top_k: Max number of results.
            similarity_threshold: Override the default threshold.

        Returns:
            List of matching CachedContext entries, sorted by relevance.
        """
        threshold = similarity_threshold or self._similarity_threshold

        async with self._lock:
            if self._index.ntotal == 0:
                self._metrics.increment("cache_miss")
                return []

            query = query_embedding.reshape(1, -1).astype(np.float32).copy()
            faiss.normalize_L2(query)

            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(query, k)

            results: list[CachedContext] = []
            now = time.time()
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx >= len(self._entries):
                    continue
                entry = self._entries[idx]
                if entry.is_expired:
                    continue
                if score >= threshold:
                    entry.access_count += 1
                    entry.last_accessed = now
                    results.append(entry)

            if results:
                self._metrics.increment("cache_hit")
                logger.debug(f"Cache hit: {len(results)} results (best score: {scores[0][0]:.3f})")
            else:
                self._metrics.increment("cache_miss")
                logger.debug(f"Cache miss (best score: {scores[0][0]:.3f} < threshold {threshold})")

            return sorted(results, key=lambda e: e.relevance_score, reverse=True)

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._entries.clear()
            self._index = faiss.IndexFlatIP(self._dimension)

    async def clear_stale(self, max_age: float | None = None) -> int:
        """Remove expired entries and return count removed."""
        async with self._lock:
            return self._evict_expired(max_age)

    def _evict_expired(self, max_age: float | None = None) -> int:
        """Remove expired entries. Must be called under lock."""
        now = time.time()
        to_keep: list[int] = []
        removed = 0

        for i, entry in enumerate(self._entries):
            expired = entry.is_expired
            if max_age is not None:
                expired = expired or (now - entry.created_at) > max_age
            if not expired:
                to_keep.append(i)
            else:
                removed += 1

        if removed > 0:
            self._rebuild_index(to_keep)
            logger.debug(f"Evicted {removed} expired entries")

        return removed

    def _evict_lru(self) -> None:
        """Remove least recently used entry. Must be called under lock."""
        if not self._entries:
            return

        # Find least recently accessed entry
        lru_idx = min(range(len(self._entries)), key=lambda i: self._entries[i].last_accessed)
        to_keep = [i for i in range(len(self._entries)) if i != lru_idx]
        self._rebuild_index(to_keep)
        logger.debug("Evicted LRU entry")

    def _rebuild_index(self, keep_indices: list[int]) -> None:
        """Rebuild the FAISS index keeping only specified entries."""
        kept_entries = [self._entries[i] for i in keep_indices]
        self._entries = kept_entries

        self._index = faiss.IndexFlatIP(self._dimension)
        if kept_entries:
            embeddings = np.stack([e.embedding for e in kept_entries]).astype(np.float32)
            faiss.normalize_L2(embeddings)
            self._index.add(embeddings)
