"""In-memory TTL Query Cache for RAG pipeline results."""
from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models import PipelineResult

_cache: dict[str, tuple[float, PipelineResult]] = {}


def _hash_query(query: str) -> str:
    """Generate MD5 hash of normalized query string."""
    clean_query = query.strip().lower()
    return hashlib.md5(clean_query.encode("utf-8")).hexdigest()


def get_cached_result(query: str) -> PipelineResult | None:
    """Check query cache for valid non-expired entry."""
    from app.config import QUERY_CACHE_ENABLED, QUERY_CACHE_TTL_SEC
    if not QUERY_CACHE_ENABLED:
        return None

    key = _hash_query(query)
    entry = _cache.get(key)
    if not entry:
        return None

    cached_time, result = entry
    if time.time() - cached_time > QUERY_CACHE_TTL_SEC:
        del _cache[key]
        return None

    return result


def cache_result(query: str, result: PipelineResult) -> None:
    """Cache pipeline result for normalized query."""
    from app.config import QUERY_CACHE_ENABLED, QUERY_CACHE_MAX_SIZE
    if not QUERY_CACHE_ENABLED:
        return

    # Don't cache error results
    if result.error:
        return

    key = _hash_query(query)

    # Evict oldest entry if max size reached
    if len(_cache) >= QUERY_CACHE_MAX_SIZE:
        oldest_key = min(_cache.keys(), key=lambda k: _cache[k][0])
        del _cache[oldest_key]

    _cache[key] = (time.time(), result)


def clear_cache() -> None:
    """Clear all cached query results."""
    _cache.clear()
