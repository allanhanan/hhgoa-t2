"""Off-topic detection guardrail using corpus centroid similarity."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.config import RELEVANCE_THRESHOLD

_corpus_centroid: NDArray[np.float32] | None = None


def set_centroid(centroid: NDArray[np.float32]) -> None:
    """Set the corpus centroid (computed during index build or at startup)."""
    global _corpus_centroid
    _corpus_centroid = centroid / (np.linalg.norm(centroid) + 1e-8)


def compute_centroid_from_vectors(vectors: NDArray) -> NDArray[np.float32]:
    """Compute the mean centroid from a set of embeddings."""
    centroid = vectors.mean(axis=0).astype(np.float32)
    return centroid / (np.linalg.norm(centroid) + 1e-8)


def is_relevant(query_embedding: NDArray[np.float32], max_passage_score: float | None = None) -> tuple[bool, float]:
    """Check if a query is relevant to the indexed corpus.

    Evaluates max top passage similarity score or corpus centroid similarity.

    Returns:
        (is_relevant, similarity_score)
    """
    if max_passage_score is not None:
        return max_passage_score >= RELEVANCE_THRESHOLD, max_passage_score

    if _corpus_centroid is None:
        return True, 1.0

    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    similarity = float(np.dot(query_norm, _corpus_centroid))

    return similarity >= RELEVANCE_THRESHOLD, similarity
