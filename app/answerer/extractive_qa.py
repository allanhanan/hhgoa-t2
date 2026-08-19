"""ONNX-powered extractive QA model for span extraction.

Tier 2 of the 3-tier cascade: runs a single forward pass through
deepset/minilm-uncased-squad2 to predict start/end positions of the
answer span within a retrieved passage.

No autoregressive decoding. No token generation loop. Single forward pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_qa_session = None
_qa_tokenizer = None

QA_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "onnx"
QA_ONNX_PATH = QA_MODEL_DIR / "minilm_qa.onnx"
QA_TOKENIZER_NAME = "deepset/minilm-uncased-squad2"


@dataclass
class AnswerResult:
    """Result from extractive QA."""
    text: str
    confidence: float
    source_passage_idx: int = 0
    start_char: int = 0
    end_char: int = 0


def _load_qa():
    """Lazy-load the ONNX QA session and tokenizer."""
    global _qa_session, _qa_tokenizer
    if _qa_session is not None:
        return _qa_session, _qa_tokenizer

    import onnxruntime as ort
    from transformers import AutoTokenizer

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    _qa_session = ort.InferenceSession(
        str(QA_ONNX_PATH), sess_opts, providers=["CPUExecutionProvider"]
    )
    _qa_tokenizer = AutoTokenizer.from_pretrained(QA_TOKENIZER_NAME)
    return _qa_session, _qa_tokenizer


def _extract_span(
    session,
    tokenizer,
    question: str,
    passage: str,
) -> tuple[str, float]:
    """Run a single QA forward pass and return (answer_text, confidence)."""
    inputs = tokenizer(
        question,
        passage,
        return_tensors="np",
        truncation=True,
        max_length=384,
        return_offsets_mapping=True,
    )

    offset_mapping = inputs.pop("offset_mapping")[0]

    feed = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        "token_type_ids": inputs["token_type_ids"].astype(np.int64),
    }

    start_logits, end_logits = session.run(None, feed)
    start_logits = start_logits[0]
    end_logits = end_logits[0]

    token_type_ids = inputs["token_type_ids"][0]
    mask = token_type_ids == 1
    start_logits[~mask] = -1e8
    end_logits[~mask] = -1e8

    start_idx = int(np.argmax(start_logits))
    end_idx = int(np.argmax(end_logits))

    if end_idx < start_idx:
        end_idx = start_idx

    from app.config import QA_MAX_SPAN_TOKENS
    if end_idx - start_idx > QA_MAX_SPAN_TOKENS:
        end_idx = start_idx + QA_MAX_SPAN_TOKENS

    start_probs = _softmax(start_logits[mask])
    end_probs = _softmax(end_logits[mask])
    mask_indices = np.where(mask)[0]
    start_local = np.searchsorted(mask_indices, start_idx)
    end_local = np.searchsorted(mask_indices, end_idx)
    start_local = min(start_local, len(start_probs) - 1)
    end_local = min(end_local, len(end_probs) - 1)
    confidence = float(start_probs[start_local] * end_probs[end_local])

    if start_idx < len(offset_mapping) and end_idx < len(offset_mapping):
        char_start = int(offset_mapping[start_idx][0])
        char_end = int(offset_mapping[end_idx][1])
        answer = passage[char_start:char_end] if char_start < len(passage) else ""
    else:
        answer = ""

    if not answer.strip():
        answer_ids = inputs["input_ids"][0][start_idx : end_idx + 1]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    return answer.strip(), confidence


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def answer(
    question: str,
    passages: list[str],
    confidence_threshold: float = 0.01,
) -> AnswerResult:
    """Run extractive QA with early-exit over ranked passages.

    Tries each passage in order (most relevant first). Returns the first
    answer with confidence above the threshold, or the best answer found.

    Args:
        question: The user's question.
        passages: List of passage texts, sorted by relevance (best first).
        confidence_threshold: Minimum confidence to accept an answer.

    Returns:
        AnswerResult with the extracted span and confidence score.
    """
    session, tokenizer = _load_qa()

    best_answer = ""
    best_confidence = 0.0
    best_idx = 0

    for i, passage in enumerate(passages[:2]):
        if not passage.strip():
            continue

        text, conf = _extract_span(session, tokenizer, question, passage)

        if conf > best_confidence:
            best_answer = text
            best_confidence = conf
            best_idx = i

        if conf >= confidence_threshold and text:
            return AnswerResult(
                text=text,
                confidence=conf,
                source_passage_idx=i,
            )

    if best_answer:
        return AnswerResult(
            text=best_answer,
            confidence=best_confidence,
            source_passage_idx=best_idx,
        )

    return AnswerResult(
        text="I cannot determine the answer from the provided context.",
        confidence=0.0,
    )


def warmup():
    """Force-load the QA model and run a dummy inference."""
    if QA_ONNX_PATH.exists():
        result = answer("test question", ["This is a test passage for warmup."])
        return True
    return False
