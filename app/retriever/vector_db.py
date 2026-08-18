"""FAISS binary index wrapper for ultra-fast Hamming-distance search."""
from __future__ import annotations

import faiss
import numpy as np
from numpy.typing import NDArray

from app.config import INDEX_PATH, EMBEDDING_DIM

_index: faiss.IndexBinary | None = None


def load_index(path: str | None = None) -> faiss.IndexBinary:
    """Load the binary FAISS index from disk into memory."""
    global _index
    path = path or INDEX_PATH
    _index = faiss.read_index_binary(path)
    if hasattr(_index, "nprobe"):
        _index.nprobe = 16  # Sub-1ms search over 6.9M vectors
    faiss.omp_set_num_threads(1)  # Single thread eliminates OpenMP overhead for small batch-1 queries (<0.05ms)
    return _index


def get_index() -> faiss.IndexBinary:
    """Return the loaded index, loading it if needed."""
    if _index is None:
        load_index()
    return _index


def search(
    binary_query: NDArray[np.uint8],
    top_k: int = 100,
) -> tuple[NDArray[np.int32], NDArray[np.int64]]:
    """Search the binary index for nearest neighbors.

    Args:
        binary_query: Packed binary query vector [48] or [1, 48].
        top_k: Number of results to return.

    Returns:
        (distances, ids) where distances are Hamming distances.
    """
    idx = get_index()
    if binary_query.ndim == 1:
        binary_query = binary_query.reshape(1, -1)
    distances, ids = idx.search(binary_query, top_k)
    return distances[0], ids[0]


def is_loaded() -> bool:
    """Check if the index is loaded."""
    return _index is not None


def index_size() -> int:
    """Return the number of vectors in the index."""
    if _index is None:
        return 0
    return _index.ntotal
