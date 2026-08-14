"""Qdrant-based vector store for production retrieval with real network latency.

Supports both Qdrant Cloud (remote) and local Qdrant (Docker).
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from voice_optimized_rag.retrieval.vector_store import SearchResult
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("qdrant_store")


class QdrantVectorStore:
    """Qdrant-based vector store — drop-in replacement for FAISSVectorStore.

    Provides the same interface as FAISSVectorStore but backed by a real
    Qdrant instance, giving actual network latency for benchmarking.

    Args:
        dimension: Embedding vector dimension.
        url: Qdrant server URL (e.g., "https://xxx.cloud.qdrant.io").
        api_key: Qdrant API key (for cloud).
        collection_name: Name of the Qdrant collection.
    """

    def __init__(
        self,
        dimension: int,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "voice_rag",
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("Install qdrant-client: pip install qdrant-client")

        self._dimension = dimension
        self._collection = collection_name

        # Connect to Qdrant
        kwargs: dict = {"url": url, "timeout": 30}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = QdrantClient(**kwargs)

        # Create collection if it doesn't exist
        collections = [c.name for c in self._client.get_collections().collections]
        if collection_name not in collections:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
        else:
            logger.info(f"Using existing Qdrant collection: {collection_name}")

    @property
    def size(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.points_count or 0

    def add_documents(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadata: list[dict] | None = None,
        ids: list[str] | None = None,
        batch_size: int = 100,
    ) -> None:
        """Add documents with pre-computed embeddings."""
        from qdrant_client.models import PointStruct

        if len(texts) != embeddings.shape[0]:
            raise ValueError("texts and embeddings must have the same length")
        if ids is not None and len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = (embeddings / norms).astype(np.float32)

        for start in range(0, len(texts), batch_size):
            batch = []
            end = min(start + batch_size, len(texts))
            for i in range(start, end):
                meta = metadata[i] if metadata else {}
                batch.append(PointStruct(
                    id=ids[i] if ids else str(uuid4()),
                    vector=normalized[i].tolist(),
                    payload={"text": texts[i], **meta},
                ))
            self._client.upsert(
                collection_name=self._collection,
                points=batch,
            )

        logger.info(f"Added {len(texts)} documents to Qdrant (total: {self.size})")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        include_embeddings: bool = False,
    ) -> list[SearchResult]:
        """Search for similar documents via Qdrant (real network round-trip)."""
        # Normalize query
        query = query_embedding.reshape(-1).astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        results = self._client.query_points(
            collection_name=self._collection,
            query=query.tolist(),
            limit=top_k,
            with_vectors=include_embeddings,
        )

        search_results = []
        for point in results.points:
            emb = None
            if include_embeddings and point.vector:
                emb = np.array(point.vector, dtype=np.float32)

            text = point.payload.get("text", "") if point.payload else ""
            meta = {k: v for k, v in (point.payload or {}).items() if k != "text"}

            search_results.append(SearchResult(
                text=text,
                metadata=meta,
                score=point.score,
                index=hash(point.id) % (2**31),
                embedding=emb,
            ))

        return search_results

    def delete_collection(self) -> None:
        """Delete the collection (for cleanup)."""
        self._client.delete_collection(self._collection)
        logger.info(f"Deleted Qdrant collection: {self._collection}")
