"""Tests for the FAISS vector store."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from voice_optimized_rag.retrieval.vector_store import FAISSVectorStore


def _make_embeddings(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    embs = rng.randn(n, dim).astype(np.float32)
    # Normalize each row
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms


def test_add_and_search(vector_store: FAISSVectorStore, dim: int):
    """Test adding documents and searching."""
    texts = ["Hello world", "Goodbye world", "Test document"]
    embeddings = _make_embeddings(3, dim, seed=1)
    metadata = [{"id": i} for i in range(3)]

    vector_store.add_documents(texts, embeddings, metadata)
    assert vector_store.size == 3

    # Search with the first embedding
    results = vector_store.search(embeddings[0], top_k=2)
    assert len(results) == 2
    assert results[0].text == "Hello world"  # Best match = itself
    assert results[0].score > 0


def test_search_empty_store(vector_store: FAISSVectorStore, dim: int):
    """Test searching an empty store returns nothing."""
    query = _make_embeddings(1, dim)[0]
    results = vector_store.search(query, top_k=5)
    assert results == []


def test_search_top_k_capped(vector_store: FAISSVectorStore, dim: int):
    """Test that top_k is capped at store size."""
    texts = ["A", "B"]
    embeddings = _make_embeddings(2, dim)
    vector_store.add_documents(texts, embeddings)

    query = _make_embeddings(1, dim, seed=99)[0]
    results = vector_store.search(query, top_k=100)
    assert len(results) == 2


def test_save_and_load(dim: int):
    """Test persisting and loading the index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_index"

        # Create and populate
        store = FAISSVectorStore(dimension=dim)
        texts = ["Doc A", "Doc B", "Doc C"]
        embeddings = _make_embeddings(3, dim)
        metadata = [{"source": f"file{i}.txt"} for i in range(3)]
        store.add_documents(texts, embeddings, metadata)
        store.save(path)

        # Load in new instance
        loaded = FAISSVectorStore(dimension=dim, index_path=path)
        assert loaded.size == 3

        # Search should work on loaded index
        results = loaded.search(embeddings[0], top_k=1)
        assert len(results) == 1
        assert results[0].text == "Doc A"
        assert results[0].metadata == {"source": "file0.txt"}


def test_mismatched_length_raises(vector_store: FAISSVectorStore, dim: int):
    """Test that mismatched texts and embeddings raise ValueError."""
    texts = ["A", "B"]
    embeddings = _make_embeddings(3, dim)

    with pytest.raises(ValueError, match="same length"):
        vector_store.add_documents(texts, embeddings)
