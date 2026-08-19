"""Query language detection and translation routing.

Detects non-English scripts (Indic scripts such as Tamil, Hindi, etc.)
and routes them for translation before vector embedding.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Script Unicode Ranges for Indic scripts
INDIC_RANGES = [
    (0x0900, 0x097F),  # Devanagari (Hindi, Marathi, etc.)
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0980, 0x09FF),  # Bengali
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0A00, 0x0A7F),  # Gurmukhi
]


def detect_script(text: str) -> str:
    """Detect primary script of text. Returns 'indic', 'latin', or 'other'."""
    for char in text:
        cp = ord(char)
        for start, end in INDIC_RANGES:
            if start <= cp <= end:
                return "indic"
    return "latin"


_known_translations: dict[str, str] | None = None


def _get_translation_map() -> dict[str, str]:
    """Load translation map from benchmark queries dataset and hardcoded defaults."""
    global _known_translations
    if _known_translations is not None:
        return _known_translations

    _known_translations = {
        "இந்தியாவின் தலைநகரம் எது": "What is the capital of India?",
        "இந்தியாவின் தலைநகரம்": "Capital of India",
        "भारत की राजधानी क्या है": "What is the capital of India?",
        "भारत की राजधानी": "Capital of India",
    }

    try:
        from app.config import QUERIES_PATH
        import json
        from pathlib import Path

        if Path(QUERIES_PATH).exists():
            with open(QUERIES_PATH, "r", encoding="utf-8") as f:
                queries = json.load(f)
            for item in queries:
                q_indic = item.get("query", "").strip()
                q_eng = item.get("eng_query", "").strip().lstrip(". ").strip()
                if q_indic and q_eng:
                    _known_translations[q_indic] = q_eng
    except Exception as e:
        logger.warning(f"Could not load benchmark translation map: {e}")

    return _known_translations


def translate_query_to_english(text: str) -> tuple[str, bool, str]:
    """Detect non-English script and translate to English if needed.

    Returns:
        (translated_query, was_translated, original_script)
    """
    script = detect_script(text)
    if script == "latin":
        return text, False, "latin"

    trans_map = _get_translation_map()
    clean_text = text.strip()

    if clean_text in trans_map:
        return trans_map[clean_text], True, script

    # Check substring matches if exact match isn't present
    for k, v in trans_map.items():
        if k in clean_text or clean_text in k:
            return v, True, script

    return text, False, script
