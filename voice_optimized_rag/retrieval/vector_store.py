"""FAISS-based vector store for document retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("vector_store")


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    text: str
    metadata: dict
    score: float
    index: int
    embedding: np.ndarray | None = None


class FAISSVectorStore:
    """FAISS-based vector store with metadata tracking.

    Args:
        dimension: Embedding vector dimension.
        index_path: Path to load/save the index.
        simulated_latency_ms: If > 0, adds artificial delay to search() to
            simulate a production vector DB (Pinecone, Qdrant, Weaviate) where
            retrieval involves network round-trips. Useful for benchmarking.
    """

    def __init__(
        self,
        dimension: int,
        index_path: Path | None = None,
        simulated_latency_ms: float = 0,
    ) -> None:
        self._dimension = dimension
        self._index_path = index_path
        self._simulated_latency_ms = simulated_latency_ms
        self._texts: list[str] = []
        self._metadata: list[dict] = []

        if index_path and index_path.exists():
            self._load(index_path)
        else:
            self._index = faiss.IndexFlatIP(dimension)  # Inner product (cosine on normalized vecs)

    @property
    def size(self) -> int:
        return self._index.ntotal

    def add_documents(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadata: list[dict] | None = None,
    ) -> None:
        """Add documents with pre-computed embeddings.

        Args:
            texts: Document text chunks.
            embeddings: Shape (n, dimension), L2-normalized.
            metadata: Optional metadata dicts per document.
        """
        if len(texts) != embeddings.shape[0]:
            raise ValueError("texts and embeddings must have the same length")

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        self._texts.extend(texts)
        self._metadata.extend(metadata or [{} for _ in texts])
        logger.info(f"Added {len(texts)} documents (total: {self.size})")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        include_embeddings: bool = False,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query_embedding: Shape (dimension,), will be L2-normalized.
            top_k: Number of results to return.
            include_embeddings: If True, include the stored document embedding
                in each result (for caching with document-level keys).

        Returns:
            List of SearchResult sorted by descending similarity.
        """
        if self.size == 0:
            return []

        # Simulate production vector DB latency (network round-trip)
        if self._simulated_latency_ms > 0:
            import time
            time.sleep(self._simulated_latency_ms / 1000)

        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        k = min(top_k, self.size)
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            emb = None
            if include_embeddings:
                emb = self._index.reconstruct(int(idx))
            results.append(SearchResult(
                text=self._texts[idx],
                metadata=self._metadata[idx],
                score=float(score),
                index=int(idx),
                embedding=emb,
            ))
        return results

    def save(self, path: Path | None = None) -> None:
        """Persist the index and metadata to disk."""
        save_path = path or self._index_path
        if save_path is None:
            raise ValueError("No path specified for saving")

        save_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(save_path / "index.faiss"))

        import json
        with open(save_path / "metadata.json", "w") as f:
            json.dump({"texts": self._texts, "metadata": self._metadata}, f)
        logger.info(f"Saved index with {self.size} vectors to {save_path}")

    def _load(self, path: Path) -> None:
        """Load index and metadata from disk."""
        loaded_index = faiss.read_index(str(path / "index.faiss"))

        # Verify dimension matches
        if loaded_index.d != self._dimension:
            logger.warning(
                f"Index dimension mismatch: expected {self._dimension}, "
                f"got {loaded_index.d}. Creating fresh index."
            )
            self._index = faiss.IndexFlatIP(self._dimension)
            return

        self._index = loaded_index

        import json
        with open(path / "metadata.json") as f:
            data = json.load(f)
        self._texts = data["texts"]
        self._metadata = data["metadata"]
        logger.info(f"Loaded index with {self.size} vectors from {path}")
