"""Embedding provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio

import numpy as np

from voice_optimized_rag.config import VORConfig


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray] = {}
        self._max_cache_size = 1000

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""

    @abstractmethod
    async def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for a batch of texts without caching."""

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts using an LRU cache."""
        if not hasattr(self, "_cache"):
            self._cache = {}
            self._max_cache_size = 1000

        results = [None] * len(texts)
        missing_indices = []
        missing_texts = []

        for i, text in enumerate(texts):
            if text in self._cache:
                # Move to end to simulate LRU
                emb = self._cache.pop(text)
                self._cache[text] = emb
                results[i] = emb
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        if missing_texts:
            missing_embeddings = await self._compute_embeddings(missing_texts)
            for idx, text, emb in zip(missing_indices, missing_texts, missing_embeddings):
                results[idx] = emb
                if len(self._cache) >= self._max_cache_size:
                    # Remove oldest (first item in dict)
                    self._cache.pop(next(iter(self._cache)))
                self._cache[text] = emb

        return np.stack(results)

    async def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns shape (dimension,)."""
        result = await self.embed([text])
        return result[0]


class OpenAIEmbedding(EmbeddingProvider):
    """Embedding provider using OpenAI's API."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        import os

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "Install openai: pip install voice-optimized-rag[openai]"
            )

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        url = base_url or os.environ.get("OPENAI_BASE_URL")

        if url and "gateway.salesforceresearch.ai" in url:
            self._client = AsyncOpenAI(
                api_key="dummy",
                base_url=url,
                default_headers={"X-Api-Key": key},
            )
        else:
            kwargs: dict = {"api_key": key}

            if url:
                kwargs["base_url"] = url

            self._client = AsyncOpenAI(**kwargs)

        self._model = model
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    async def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )

        return np.array(
            [item.embedding for item in response.data],
            dtype=np.float32,
        )


class OllamaEmbedding(EmbeddingProvider):
    """Embedding provider using Ollama's local API."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dim: int = 768,
    ) -> None:
        super().__init__()
        import httpx

        self._client = httpx.AsyncClient(timeout=60.0)
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    async def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        response = await self._client.post(
            f"{self._base_url}/api/embed",
            json={
                "model": self._model,
                "input": texts,
            },
        )

        response.raise_for_status()

        data = response.json()

        return np.array(
            data["embeddings"],
            dtype=np.float32,
        )


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Embedding provider using a local Sentence Transformer model."""

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
    ) -> None:
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Install sentence-transformers with: "
                "uv pip install sentence-transformers"
            )

        self._model = SentenceTransformer(model)

        get_dimension = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        dimension = get_dimension()

        if dimension is None:
            raise RuntimeError(
                f"Could not determine embedding dimension for model: {model}"
            )

        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    async def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        def compute():
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(embeddings, dtype=np.float32)

        return await asyncio.to_thread(compute)


class ONNXEmbedding(EmbeddingProvider):
    """Embedding provider using int8 quantized ONNX models via Optimum for extreme speed and low RAM."""

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        super().__init__()
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("Install optimum and transformers: pip install optimum[onnxruntime] transformers")
            
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Using export=True to convert it on the fly if not already ONNX
        self._model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
        
        self._dim = self._model.config.hidden_size

    @property
    def dimension(self) -> int:
        return self._dim

    async def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        def compute():
            import torch
            import torch.nn.functional as F
            
            inputs = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = self._model(**inputs)
            
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # L2 Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.numpy().astype(np.float32)

        return await asyncio.to_thread(compute)


def create_embedding_provider(config: VORConfig) -> EmbeddingProvider:
    """Factory function to create an embedding provider from config."""

    provider = config.embedding_provider

    if provider == "openai":
        return OpenAIEmbedding(
            api_key=config.llm_api_key,
            model=config.embedding_model,
            dim=config.embedding_dimension,
            base_url=config.llm_base_url,
        )

    elif provider == "ollama":
        return OllamaEmbedding(
            model=config.embedding_model,
            base_url=config.llm_base_url or "http://localhost:11434",
            dim=config.embedding_dimension,
        )

    elif provider == "sentence-transformers":
        return SentenceTransformerEmbedding(
            model=config.embedding_model or "all-MiniLM-L6-v2",
        )
        
    elif provider == "onnx":
        # Fallback to sentence-transformers model ID format if it's not provided
        model = config.embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        # If user only specifies 'all-MiniLM-L6-v2', HF optimum needs the author prefix for some models
        if '/' not in model and model == 'all-MiniLM-L6-v2':
            model = f"sentence-transformers/{model}"
            
        return ONNXEmbedding(model_id=model)

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}"
        )
