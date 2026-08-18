"""Float16 rescoring to recover accuracy after binary quantization."""
from __future__ import annotations
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.config import FLOAT16_PATH

_float16_vectors: np.ndarray | None = None


def load_vectors(path: str | None = None) -> np.ndarray:
    """Load float16 vectors memory-mapped into RAM."""
    global _float16_vectors
    path = path or FLOAT16_PATH
    if _float16_vectors is None:
        try:
            _float16_vectors = np.load(path, mmap_mode="r")
        except Exception:
            # Handle raw memmap array (6.9M x 384)
            n_bytes = Path(path).stat().st_size
            n_rows = n_bytes // (384 * 2)  # float16 is 2 bytes
            _float16_vectors = np.memmap(path, dtype=np.float16, mode="r", shape=(n_rows, 384))
    return _float16_vectors


def get_vectors() -> NDArray[np.float16]:
    """Return loaded vectors, loading if needed."""
    if _float16_vectors is None:
        load_vectors()
    return _float16_vectors


def rescore(
    query_embedding: NDArray[np.float32],
    candidate_ids: NDArray[np.int64],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Re-rank binary search candidates using precise dot-product scoring.

    Args:
        query_embedding: Float32 query embedding [384].
        candidate_ids: IDs from binary search [N].
        top_k: Number of final results after re-ranking.

    Returns:
        List of (vector_id, score) tuples, sorted by score descending.
    """
    vectors = get_vectors()

    # Filter out invalid IDs (-1 means no result from FAISS)
    valid_mask = candidate_ids >= 0
    valid_ids = candidate_ids[valid_mask]

    if len(valid_ids) == 0:
        return []

    # Limit to top 8 candidates to avoid random 5GB disk page faults
    if len(valid_ids) > 8:
        valid_ids = valid_ids[:8]

    # Fetch float16 vectors for candidates
    try:
        candidate_vecs = np.take(vectors, valid_ids, axis=0).astype(np.float32)
        scores = candidate_vecs @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(valid_ids[i]), float(scores[i])) for i in top_indices]
    except Exception:
        # Fallback: rank by candidate order (already sorted by FAISS binary Hamming distance)
        return [(int(valid_ids[i]), float(1.0 - (i * 0.01))) for i in range(min(len(valid_ids), top_k))]


def is_loaded() -> bool:
    """Check if vectors are loaded."""
    return _float16_vectors is not None
