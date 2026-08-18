"""Output grounding check — verify LLM answer is grounded in retrieved context."""
from __future__ import annotations

import re

from app.config import GROUNDING_OVERLAP_THRESHOLD


def _tokenize(text: str) -> set[str]:
    """Simple whitespace + punctuation tokenizer."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def check_grounding(answer: str, passages: list[str]) -> tuple[bool, float]:
    """Check if the LLM's answer is grounded in the retrieved passages.

    Uses token-level overlap ratio: what fraction of answer tokens
    appear in the concatenated passage text.

    Returns:
        (is_grounded, overlap_ratio)
    """
    if not answer or not passages:
        return True, 1.0  # Nothing to check

    # "I cannot answer" responses are always grounded
    refusal_phrases = [
        "cannot answer",
        "don't have enough",
        "not in the context",
        "no relevant information",
        "not enough information",
    ]
    answer_lower = answer.lower()
    for phrase in refusal_phrases:
        if phrase in answer_lower:
            return True, 1.0

    answer_tokens = _tokenize(answer)
    # Remove common stop words from overlap check
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "it", "its", "this", "that", "and", "or", "but", "not", "no",
        "if", "then", "than", "so", "as",
    }
    answer_tokens -= stop_words

    if not answer_tokens:
        return True, 1.0

    passage_tokens = set()
    for p in passages:
        passage_tokens |= _tokenize(p)

    overlap = answer_tokens & passage_tokens
    ratio = len(overlap) / len(answer_tokens)

    return ratio >= GROUNDING_OVERLAP_THRESHOLD, ratio
