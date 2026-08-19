"""Heuristic fast-path answer extraction for high-confidence retrievals.

Tier 1 of the 3-tier cascade: uses question-type classification and
regex-based entity extraction to answer simple factoid queries in <1ms
without any neural inference.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class HeuristicResult:
    """Result from heuristic extraction."""
    answer: str | None
    confidence: float
    method: str


# ── Question type patterns ────────────────────────────────────────────

_Q_PATTERNS = {
    "PERSON": re.compile(
        r"^(who|whom)\b", re.IGNORECASE
    ),
    "DATE": re.compile(
        r"^(when|what year|what date|which year|which date)\b", re.IGNORECASE
    ),
    "LOCATION": re.compile(
        r"^(where|what country|what city|what state|which country|which city)\b",
        re.IGNORECASE,
    ),
    "NUMBER": re.compile(
        r"^(how many|how much|how old|how long|how far|how tall|how high|what percentage|what number)\b",
        re.IGNORECASE,
    ),
    "DEFINITION": re.compile(
        r"^(what is|what are|what was|what were|define|meaning of)\b",
        re.IGNORECASE,
    ),
}

# ── Entity extraction patterns ────────────────────────────────────────

_ENTITY_PATTERNS = {
    "PERSON": re.compile(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"
    ),
    "DATE": re.compile(
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}|\d{4}|"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
    "NUMBER": re.compile(
        r"\b(\d[\d,]*\.?\d*\s*(?:%|percent|million|billion|trillion|thousand|"
        r"km|miles|meters|feet|inches|pounds|kg|years|months|days|hours)?)\b",
        re.IGNORECASE,
    ),
    "LOCATION": re.compile(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"
    ),
}


def classify_question(question: str) -> str | None:
    """Classify a question into a type: PERSON, DATE, LOCATION, NUMBER, DEFINITION."""
    question = question.strip()
    for q_type, pattern in _Q_PATTERNS.items():
        if pattern.match(question):
            return q_type
    return None


def extract_candidates(passage: str, q_type: str) -> list[str]:
    """Extract candidate answer entities from a passage based on question type."""
    pattern = _ENTITY_PATTERNS.get(q_type)
    if pattern is None:
        return []

    matches = pattern.findall(passage)
    seen = set()
    unique = []
    for m in matches:
        m_stripped = m.strip()
        if m_stripped and m_stripped.lower() not in seen:
            seen.add(m_stripped.lower())
            unique.append(m_stripped)
    return unique


def heuristic_extract(question: str, passage: str) -> HeuristicResult:
    """Try to extract an answer using lightweight heuristics.

    Returns HeuristicResult with answer=None if heuristic cannot confidently answer.
    """
    q_type = classify_question(question)
    if q_type is None:
        return HeuristicResult(answer=None, confidence=0.0, method="no_match")

    if q_type == "DEFINITION":
        q_clean = re.sub(r"^(what is|what are|what was|what were|define|meaning of)\s+", "", question, flags=re.IGNORECASE)
        q_clean = q_clean.rstrip("?").strip()
        if q_clean:
            def_pattern = re.compile(
                rf"\b{re.escape(q_clean)}\b\s+(?:is|are|was|were|refers?\s+to)\s+(.{{10,150}}?)(?:\.|$)",
                re.IGNORECASE,
            )
            match = def_pattern.search(passage)
            if match:
                answer = f"{q_clean} {match.group(0).split(q_clean, 1)[-1].strip()}"
                return HeuristicResult(answer=answer.strip(". "), confidence=0.75, method="definition_pattern")
        return HeuristicResult(answer=None, confidence=0.0, method="definition_no_match")

    candidates = extract_candidates(passage, q_type)
    if len(candidates) == 1:
        return HeuristicResult(answer=candidates[0], confidence=0.8, method=f"single_{q_type.lower()}")
    elif len(candidates) == 0:
        return HeuristicResult(answer=None, confidence=0.0, method=f"no_{q_type.lower()}")
    else:
        return HeuristicResult(answer=None, confidence=0.3, method=f"multiple_{q_type.lower()}")
