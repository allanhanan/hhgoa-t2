"""Embedder module exposing embed, embed_one, and get_model for eval loop and pipeline."""
from __future__ import annotations

import numpy as np
from app.embedder.encoder import encode, encode_batch, warmup


def embed_one(text: str) -> np.ndarray:
    """Embed a single string into a 1D float32 vector (shape: [dim])."""
    return encode(text)


def embed(texts: list[str]) -> np.ndarray:
    """Embed a batch of strings into a 2D float32 array (shape: [N, dim])."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return encode_batch(texts)


def get_model():
    """Ensure the embedding model is loaded and warmed up."""
    warmup()
    return "minilm-onnx"
