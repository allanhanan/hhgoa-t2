from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class GPUEncoder:
    def __init__(self, batch_size: int = 128, device: str | None = None):
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {MODEL_NAME} on {self.device}...")

        self.model = SentenceTransformer(
            MODEL_NAME,
            device=self.device,
        )

        self.model.eval()

        if torch.cuda.is_available() and self.device == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
        print("Embedding dimension:", self.model.get_sentence_embedding_dimension())

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.astype(np.float32)
