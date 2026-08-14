"""Shared test fixtures."""

from __future__ import annotations

from typing import AsyncIterator

import numpy as np
import pytest

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.semantic_cache import SemanticCache
from voice_optimized_rag.core.conversation_stream import ConversationStream
from voice_optimized_rag.llm.base import LLMProvider
from voice_optimized_rag.retrieval.embeddings import EmbeddingProvider
from voice_optimized_rag.retrieval.vector_store import FAISSVectorStore
from voice_optimized_rag.utils.metrics import MetricsCollector


# --- Mock Providers ---

class MockLLM(LLMProvider):
    """Mock LLM that returns predictable responses."""

    def __init__(self, response: str = "Mock response") -> None:
        self._response = response
        self.call_count = 0
        self.last_prompt = ""
        self.last_context = ""

    async def generate(self, prompt: str, context: str = "") -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_context = context
        if "predict" in prompt.lower() and "follow-up" in prompt.lower():
            return "1. What is the pricing?\n2. How does billing work?\n3. Are there discounts?"
        if "keywords" in prompt.lower():
            return "pricing, billing, discounts"
        return self._response

    async def stream(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_context = context
        for word in self._response.split():
            yield word + " "


class MockEmbedding(EmbeddingProvider):
    """Mock embedding provider that returns deterministic embeddings."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    async def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        # Generate deterministic embeddings based on text hash
        rng = np.random.RandomState(42)
        embeddings = []
        for text in texts:
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)


# --- Fixtures ---

@pytest.fixture
def dim() -> int:
    return 64


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def mock_embeddings(dim: int) -> MockEmbedding:
    return MockEmbedding(dim=dim)


@pytest.fixture
def metrics() -> MetricsCollector:
    return MetricsCollector()


@pytest.fixture
def cache(dim: int, metrics: MetricsCollector) -> SemanticCache:
    return SemanticCache(
        dimension=dim,
        max_size=100,
        default_ttl=60.0,
        similarity_threshold=0.5,
        metrics=metrics,
    )


@pytest.fixture
def vector_store(dim: int) -> FAISSVectorStore:
    return FAISSVectorStore(dimension=dim)


@pytest.fixture
def stream() -> ConversationStream:
    return ConversationStream(window_size=10)


@pytest.fixture
def config() -> VORConfig:
    return VORConfig(
        llm_provider="openai",
        llm_api_key="test-key",
        embedding_dimension=64,
        cache_max_size=100,
        cache_ttl_seconds=60.0,
        cache_similarity_threshold=0.5,
        vector_store_provider="faiss",
        guardrail_min_relevance_score=-1.0,
    )
