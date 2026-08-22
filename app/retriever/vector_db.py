"""FAISS index wrapper supporting both IndexIVFScalarQuantizer (SQ8) and IndexBinaryFlat (fallback)."""
from __future__ import annotations
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from app.config import INDEX_PATH, IVF_INDEX_PATH, USE_IVF, IVF_NPROBE, ANN_TOP_K, TOP_K_BINARY, EMBEDDING_DIM, FAISS_THREADS

_index: faiss.Index | faiss.IndexBinary | None = None
_is_ivf: bool = False
_gpu_res = None  # Keep reference alive to prevent GPU resource deallocation


def _try_gpu_index(index):
    """Move FAISS index to GPU if faiss-gpu is installed and DEVICE != 'cpu'.

    Only beneficial for IVF indices. BinaryFlat uses CPU POPCNT which is
    already hardware-optimal for Hamming distance.
    """
    global _gpu_res
    from app.config import DEVICE
    if DEVICE == "cpu":
        return index
    try:
        _gpu_res = faiss.StandardGpuResources()
        _gpu_res.setTempMemory(64 * 1024 * 1024)  # 64MB GPU temp memory
        gpu_index = faiss.index_cpu_to_gpu(_gpu_res, 0, index)
        return gpu_index
    except (AttributeError, RuntimeError):
        # faiss-gpu not installed or no GPU available — use CPU index
        return index


def load_index(path: str | None = None, use_ivf: bool | None = None) -> faiss.Index | faiss.IndexBinary:
    """Load FAISS index (IndexIVFScalarQuantizer SQ8 ANN index or BinaryFlat fallback)."""
    global _index, _is_ivf
    use_ivf = USE_IVF if use_ivf is None else use_ivf

    # If index is already loaded, return cached singleton instance to prevent duplicate memory allocation
    if _index is not None:
        return _index

    if use_ivf:
        ivf_path = path or IVF_INDEX_PATH
        if Path(ivf_path).exists():
            try:
                _index = faiss.read_index(ivf_path, faiss.IO_FLAG_READ_ONLY)
                _index = _try_gpu_index(_index)
                _is_ivf = True
                if hasattr(_index, "nprobe"):
                    _index.nprobe = IVF_NPROBE
                faiss.omp_set_num_threads(FAISS_THREADS)
                prewarm_index()
                return _index
            except Exception:
                pass  # Fallback to binary index if memory allocation fails

    path = path or INDEX_PATH
    _index = faiss.read_index_binary(path)
    _is_ivf = False
    if hasattr(_index, "nprobe"):
        _index.nprobe = 16
    faiss.omp_set_num_threads(FAISS_THREADS)
    prewarm_index()
    return _index


def prewarm_index() -> None:
    """Pre-warm loaded index memory pages by running dummy searches."""
    if _index is None:
        return
    dummy_vec = (
        np.zeros((1, 384), dtype=np.float32)
        if _is_ivf
        else np.zeros((1, 48), dtype=np.uint8)
    )
    for _ in range(5):
        try:
            _index.search(dummy_vec, 10)
        except Exception:
            pass



def get_index() -> faiss.Index | faiss.IndexBinary:
    """Return loaded index, loading default if needed."""
    if _index is None:
        load_index()
    return _index


def is_ivf() -> bool:
    """Check if the currently loaded index is an IVF ANN index."""
    if _index is None:
        load_index()
    return _is_ivf


def set_nprobe(nprobe: int) -> None:
    """Dynamically set nprobe parameter for IVF search."""
    idx = get_index()
    if hasattr(idx, "nprobe"):
        idx.nprobe = nprobe


def search(
    query: NDArray[np.float32] | NDArray[np.uint8],
    top_k: int | None = None,
) -> tuple[NDArray[np.float32] | NDArray[np.int32], NDArray[np.int64]]:
    """Search nearest neighbors. Accepts float32 vector for IVF or uint8 packed binary vector for Binary search.

    Returns:
        (distances, ids)
    """
    idx = get_index()
    if _is_ivf:
        k = top_k if top_k is not None else ANN_TOP_K
        if query.dtype != np.float32 or query.ndim == 1:
            q_float = query.astype(np.float32).reshape(1, -1)
        else:
            q_float = query.reshape(1, -1)
        distances, ids = idx.search(q_float, k)
    else:
        k = top_k if top_k is not None else TOP_K_BINARY
        if query.ndim == 1:
            query = query.reshape(1, -1)
        distances, ids = idx.search(query, k)

    return distances[0], ids[0]


def is_loaded() -> bool:
    """Check if index is loaded."""
    return _index is not None


def index_size() -> int:
    """Return total vectors in index."""
    if _index is None:
        return 0
    return _index.ntotal
