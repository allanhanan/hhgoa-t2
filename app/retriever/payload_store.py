"""SQLite payload store for fast passage text lookup by vector ID."""
from __future__ import annotations

import sqlite3
from typing import Any

from app.config import PAYLOAD_DB
from app.models import PassageResult

_conn: sqlite3.Connection | None = None


def connect(path: str | None = None) -> sqlite3.Connection:
    """Open a read-only SQLite connection with performance PRAGMAs."""
    global _conn
    path = path or PAYLOAD_DB
    _conn = sqlite3.connect(path, check_same_thread=False)
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA synchronous=OFF")
    _conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
    _conn.row_factory = sqlite3.Row
    return _conn


def get_conn() -> sqlite3.Connection:
    """Return existing connection, creating if needed."""
    if _conn is None:
        connect()
    return _conn


def fetch(ids: list[int]) -> list[PassageResult]:
    """Fetch passage payloads by vector IDs.

    Args:
        ids: List of vector IDs (matching FAISS index positions).

    Returns:
        List of PassageResult objects with text and metadata.
    """
    if not ids:
        return []

    conn = get_conn()
    placeholders = ",".join("?" for _ in ids)
    cursor = conn.execute(
        f"SELECT id, text, query_type, source_query FROM payloads WHERE id IN ({placeholders})",
        ids,
    )

    # Build a lookup dict to preserve score ordering from caller
    rows = {row["id"]: row for row in cursor.fetchall()}

    results = []
    for vid in ids:
        if vid in rows:
            row = rows[vid]
            results.append(PassageResult(
                id=row["id"],
                text=row["text"],
                score=0.0,  # Score set by caller
                query_type=row["query_type"] or "",
                source_query=row["source_query"] or "",
            ))
    return results


def is_loaded() -> bool:
    """Check if the database connection is open."""
    return _conn is not None
