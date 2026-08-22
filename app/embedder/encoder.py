"""ONNX-accelerated sentence encoder with binary quantization."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_onnx_session = None
_tokenizer = None
_model = None

ONNX_INT8_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "onnx" / "minilm_int8.onnx"
ONNX_FP32_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "onnx" / "minilm_fp32.onnx"
ONNX_MODEL_PATH = ONNX_INT8_PATH if ONNX_INT8_PATH.exists() else ONNX_FP32_PATH


def _load_onnx_or_fallback():
    """Load ONNX Runtime session if model file exists, else fallback to sentence-transformers."""
    global _onnx_session, _tokenizer, _model
    if _onnx_session is not None:
        return ("onnx", _onnx_session, _tokenizer)
    if _model is not None:
        return ("pt", _model, None)

    if ONNX_MODEL_PATH.exists():
        import onnxruntime as ort
        from transformers import AutoTokenizer

        from app.config import get_onnx_providers, get_onnx_session_options

        sess_opts = get_onnx_session_options(ONNX_MODEL_PATH)
        providers = get_onnx_providers(ONNX_MODEL_PATH)
        _onnx_session = ort.InferenceSession(str(ONNX_MODEL_PATH), sess_opts, providers=providers)
        _tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", use_fast=True)
        return ("onnx", _onnx_session, _tokenizer)

    try:
        from sentence_transformers import SentenceTransformer
        from app.config import EMBEDDING_MODEL
        _model = SentenceTransformer(EMBEDDING_MODEL)
        return ("pt", _model, None)
    except ImportError:
        raise RuntimeError(
            f"ONNX model not found at {ONNX_MODEL_PATH} and 'sentence-transformers' is not installed. "
            "Please ensure the ONNX model exists in models/onnx/ or install sentence-transformers."
        )


def encode(text: str) -> NDArray[np.float32]:
    """Encode a single query string → float32 embedding [384]."""
    kind, sess_or_model, tok = _load_onnx_or_fallback()
    from app.config import MAX_TOKEN_LIMIT
    if kind == "onnx":
        inputs = tok(text, return_tensors="np", truncation=True, max_length=MAX_TOKEN_LIMIT)
        feed = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"])),
        }
        out = sess_or_model.run(None, feed)[0]
        # Mean pooling + L2 norm
        mask = feed["attention_mask"][:, :, None]
        emb = (out * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = (emb / np.maximum(norm, 1e-9)).astype(np.float32)
        return emb[0]

    return sess_or_model.encode(text, normalize_embeddings=True).astype(np.float32)


def encode_batch(texts: list[str], batch_size: int = 256) -> NDArray[np.float32]:
    """Encode a batch of texts → float32 embeddings [N, 384]."""
    kind, sess_or_model, tok = _load_onnx_or_fallback()
    if kind == "onnx":
        from app.config import MAX_TOKEN_LIMIT
        res = []
        for i in range(0, len(texts), batch_size):
            b_texts = texts[i : i + batch_size]
            inputs = tok(b_texts, return_tensors="np", padding=True, truncation=True, max_length=MAX_TOKEN_LIMIT)
            inputs_onnx = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
                "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"])).astype(np.int64),
            }
            out = sess_or_model.run(None, inputs_onnx)[0]
            mask = inputs_onnx["attention_mask"][:, :, None]
            emb = (out * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = (emb / np.maximum(norm, 1e-9)).astype(np.float32)
            res.append(emb)
        return np.vstack(res)

    return sess_or_model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


def binarize(embedding: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Convert float32 embedding to packed binary (384 bits → 48 bytes)."""
    if embedding.ndim == 1:
        bits = (embedding > 0).astype(np.uint8)
        return np.packbits(bits)
    bits = (embedding > 0).astype(np.uint8)
    return np.packbits(bits, axis=1)


def encode_and_binarize(text: str) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """Encode + binarize in one call. Returns (float32_emb, binary_emb)."""
    emb = encode(text)
    return emb, binarize(emb)


def warmup():
    """Force model load and run multiple dummy inferences to warm ONNX execution graph."""
    encode("warmup query short")
    encode("What is retrieval augmented generation and how does vector search work?")

