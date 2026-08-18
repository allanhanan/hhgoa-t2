"""Input safety guardrail — keyword blocklist and input sanitization."""
from __future__ import annotations

import re

from app.config import SAFETY_MAX_QUERY_LEN

# Blocklist of unsafe/inappropriate terms
_BLOCKED_PATTERNS = [
    r"\b(hack|exploit|inject|drop\s+table|delete\s+from)\b",
    r"\b(kill|murder|bomb|weapon|terrorist)\b",
    r"\b(porn|xxx|nsfw|nude)\b",
    r"<script|javascript:|on\w+=",  # XSS patterns
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _BLOCKED_PATTERNS]


def check_safety(query: str) -> tuple[bool, str]:
    """Check if a query is safe to process.

    Returns:
        (is_safe, reason) — if not safe, reason explains why.
    """
    # Length check
    if not query or not query.strip():
        return False, "Empty query"

    if len(query) > SAFETY_MAX_QUERY_LEN:
        return False, f"Query too long ({len(query)} chars, max {SAFETY_MAX_QUERY_LEN})"

    # Blocklist check
    for pattern in _COMPILED:
        if pattern.search(query):
            return False, "Query contains blocked content"

    return True, ""
