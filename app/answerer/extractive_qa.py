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

from app.config import QA_ONNX_MODEL, QA_TOKENIZER_NAME

_qa_session = None
_qa_tokenizer = None
_qa_cache: dict[tuple[str, str], AnswerResult] = {}
_MAX_QA_CACHE_SIZE = 5000

QA_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "onnx"
QA_ONNX_PATH = Path(QA_ONNX_MODEL) if Path(QA_ONNX_MODEL).exists() else (QA_MODEL_DIR / "minilm_qa_int8.onnx")


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
    sess_opts.inter_op_num_threads = 1
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    _qa_session = ort.InferenceSession(
        str(QA_ONNX_PATH), sess_opts, providers=["CPUExecutionProvider"]
    )
    _qa_tokenizer = AutoTokenizer.from_pretrained(QA_TOKENIZER_NAME, use_fast=True)
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
        max_length=128,
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
    context_mask = token_type_ids == 1

    # Only suppress question/SEP tokens — CLS (index 0) stays eligible,
    # since it's the model's "no answer" signal.
    suppress_mask = ~context_mask
    suppress_mask[0] = False

    start_logits = start_logits.copy()
    end_logits = end_logits.copy()
    start_logits[suppress_mask] = -1e8
    end_logits[suppress_mask] = -1e8

    # Null (no-answer) score
    null_score = float(start_logits[0] + end_logits[0])

    # Best non-null span, searched only within context tokens
    ctx_start = np.where(context_mask, start_logits, -1e8)
    ctx_end = np.where(context_mask, end_logits, -1e8)
    start_idx = int(np.argmax(ctx_start))
    end_idx = int(np.argmax(ctx_end))

    if end_idx < start_idx:
        end_idx = start_idx

    from app.config import QA_MAX_SPAN_TOKENS
    if end_idx - start_idx > QA_MAX_SPAN_TOKENS:
        end_idx = start_idx + QA_MAX_SPAN_TOKENS

    best_non_null_score = float(ctx_start[start_idx] + ctx_end[end_idx])

    # SQuAD2 convention: predict null if it beats the best span by
    # more than NULL_THRESHOLD.
    from app.config import QA_NULL_THRESHOLD
    if null_score > best_non_null_score - QA_NULL_THRESHOLD:
        return "", 0.0  # explicit "no answer" signal

    start_probs = _softmax(ctx_start[context_mask])
    end_probs = _softmax(ctx_end[context_mask])
    mask_indices = np.where(context_mask)[0]
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
    confidence_threshold: float | None = None,
) -> AnswerResult:
    """Run extractive QA with early-exit over ranked passages.

    Tries each passage in order (most relevant first). Returns the first
    answer with confidence above the threshold and sufficient margin,
    or an empty result.

    Args:
        question: The user's question.
        passages: List of passage texts, sorted by relevance (best first).
        confidence_threshold: Minimum confidence to accept an answer.

    Returns:
        AnswerResult with the extracted span and confidence score.
    """
    from app.config import QA_CONFIDENCE_THRESHOLD, QA_MARGIN_THRESHOLD
    if confidence_threshold is None:
        confidence_threshold = QA_CONFIDENCE_THRESHOLD

    top_p = passages[0] if passages else ""
    cache_key = (question.strip().lower(), top_p)
    if cache_key in _qa_cache:
        return _qa_cache[cache_key]

    session, tokenizer = _load_qa()

    best_answer = ""
    best_confidence = 0.0
    second_best_confidence = 0.0
    best_idx = 0

    for i, passage in enumerate(passages[:1]):
        if not passage.strip():
            continue

        text, conf = _extract_span(session, tokenizer, question, passage)
        if not text:
            continue

        if conf > best_confidence:
            second_best_confidence = best_confidence
            best_answer, best_confidence, best_idx = text, conf, i

        if best_confidence >= confidence_threshold:
            break




    margin_ok = (best_confidence - second_best_confidence) >= QA_MARGIN_THRESHOLD
    if best_answer and best_confidence >= confidence_threshold and margin_ok:
        res = AnswerResult(
            text=best_answer,
            confidence=best_confidence,
            source_passage_idx=best_idx,
        )
    else:
        res = AnswerResult(
            text="",
            confidence=0.0,
        )

    if len(_qa_cache) < _MAX_QA_CACHE_SIZE:
        _qa_cache[cache_key] = res
    return res


def clear_qa_cache():
    """Clear in-memory QA cache."""
    _qa_cache.clear()


def warmup():
    """Force-load the QA model and run dummy inferences for fast and fallback branches."""
    if QA_ONNX_PATH.exists():
        answer("What is the capital of France?", ["Paris is the capital of France."])
        answer("Unrelated question?", ["Photosynthesis happens in plants."])
        return True
    return False

