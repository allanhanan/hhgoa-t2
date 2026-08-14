"""Memory Router: Central orchestrator for the dual-agent architecture.

This is the main entry point for using VoiceAgentRAG. It initializes
and manages the Slow Thinker, Fast Talker, semantic cache, vector store,
and conversation stream.
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

import numpy as np

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.conversation_stream import (
    ConversationStream,
    EventType,
    StreamEvent,
)
from voice_optimized_rag.core.fast_talker import FastTalker
from voice_optimized_rag.core.semantic_cache import SemanticCache
from voice_optimized_rag.core.slow_thinker import SlowThinker
from voice_optimized_rag.llm.base import LLMProvider, create_llm
from voice_optimized_rag.retrieval.document_loader import DocumentChunk, load_directory
from voice_optimized_rag.retrieval.embeddings import EmbeddingProvider, create_embedding_provider
from voice_optimized_rag.retrieval.vector_store import FAISSVectorStore
from voice_optimized_rag.utils.logging import get_logger, setup_logging
from voice_optimized_rag.utils.metrics import MetricsCollector

logger = get_logger("memory_router")


class MemoryRouter:
    """Central orchestrator for the VoiceAgentRAG dual-agent system.

    Usage:
        config = VORConfig(llm_provider="openai", llm_api_key="sk-...")
        router = MemoryRouter(config)
        await router.start()

        # Ingest documents
        await router.ingest_directory(Path("docs/"))

        # Query
        response = await router.query("What is the pricing?")
        print(response)

        # Or stream
        async for chunk in router.query_stream("Tell me about features"):
            print(chunk, end="", flush=True)

        await router.stop()
    """

    def __init__(
        self,
        config: VORConfig | None = None,
        llm: LLMProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._config = config or VORConfig()
        self._metrics = MetricsCollector()

        # Initialize components
        self._llm = llm or create_llm(self._config)
        self._embeddings = embedding_provider or create_embedding_provider(self._config)

        # Create vector store (FAISS or Qdrant)
        if self._config.vector_store_provider == "qdrant":
            from voice_optimized_rag.retrieval.qdrant_store import QdrantVectorStore
            self._vector_store = QdrantVectorStore(
                dimension=self._embeddings.dimension,
                url=self._config.qdrant_url,
                api_key=self._config.qdrant_api_key or None,
                collection_name=self._config.qdrant_collection,
            )
        else:
            index_path = self._config.faiss_index_path
            load_path = index_path if (index_path.exists() and (index_path / "index.faiss").exists()) else None
            self._vector_store = FAISSVectorStore(
                dimension=self._embeddings.dimension,
                index_path=load_path,
                simulated_latency_ms=self._config.retrieval_latency_ms,
            )
        self._cache = SemanticCache(
            dimension=self._embeddings.dimension,
            max_size=self._config.cache_max_size,
            default_ttl=self._config.cache_ttl_seconds,
            similarity_threshold=self._config.cache_similarity_threshold,
            metrics=self._metrics,
        )
        self._stream = ConversationStream(
            window_size=self._config.conversation_window_size,
        )

        # Initialize agents
        self._slow_thinker = SlowThinker(
            config=self._config,
            llm=self._llm,
            embedding_provider=self._embeddings,
            vector_store=self._vector_store,
            cache=self._cache,
            stream=self._stream,
            metrics=self._metrics,
        )
        self._fast_talker = FastTalker(
            config=self._config,
            llm=self._llm,
            embedding_provider=self._embeddings,
            vector_store=self._vector_store,
            cache=self._cache,
            stream=self._stream,
            metrics=self._metrics,
        )
        self._running = False

    async def start(self, log_level: str = "INFO") -> None:
        """Start the memory router and background agents."""
        setup_logging(log_level)
        await self._slow_thinker.start()
        self._running = True
        logger.info("Memory Router started")

    async def stop(self) -> None:
        """Stop the memory router and all agents."""
        self._running = False
        await self._slow_thinker.stop()
        logger.info("Memory Router stopped")

    async def query(self, text: str) -> str:
        """Process a user query and return a complete response.

        Args:
            text: The user's question or statement.

        Returns:
            The generated response text.
        """
        # Publish user utterance to stream (triggers slow thinker)
        await self._stream.publish(StreamEvent(
            event_type=EventType.USER_UTTERANCE,
            text=text,
        ))

        # Get response from fast talker
        response = await self._fast_talker.respond(text)

        # Publish agent response to stream
        await self._stream.publish(StreamEvent(
            event_type=EventType.AGENT_RESPONSE,
            text=response,
        ))

        return response

    async def query_stream(self, text: str) -> AsyncIterator[str]:
        """Process a user query and stream the response.

        Args:
            text: The user's question or statement.

        Yields:
            Response text chunks.
        """
        await self._stream.publish(StreamEvent(
            event_type=EventType.USER_UTTERANCE,
            text=text,
        ))

        full_response: list[str] = []
        async for chunk in self._fast_talker.respond_stream(text):
            full_response.append(chunk)
            yield chunk

        await self._stream.publish(StreamEvent(
            event_type=EventType.AGENT_RESPONSE,
            text="".join(full_response),
        ))

    async def ingest_directory(
        self,
        directory: Path,
        extensions: set[str] | None = None,
    ) -> int:
        """Ingest all documents from a directory into the vector store.

        Args:
            directory: Path to the document directory.
            extensions: File extensions to include.

        Returns:
            Number of chunks ingested.
        """
        chunks = load_directory(
            directory,
            extensions=extensions,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            chunking_strategy=self._config.chunking_strategy,
            parent_chunk_size=self._config.parent_chunk_size,
        )
        return await self._ingest_chunks(chunks)

    async def ingest_texts(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> int:
        """Ingest raw text strings into the vector store.

        Args:
            texts: List of text strings to ingest.
            metadata: Optional metadata per text.

        Returns:
            Number of chunks ingested.
        """
        chunks = [
            DocumentChunk(text=t, metadata=m)
            for t, m in zip(texts, metadata or [{} for _ in texts])
        ]
        return await self._ingest_chunks(chunks)

    async def _ingest_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embed and store document chunks."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        metadata = [c.metadata for c in chunks]

        # Batch embed
        batch_size = 100
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            with self._metrics_timer("ingestion", "embedding"):
                embeddings = await self._embeddings.embed(batch)
            all_embeddings.append(embeddings)

        embeddings_array = np.vstack(all_embeddings)
        with self._metrics_timer("ingestion", "retrieval"):
            self._vector_store.add_documents(texts, embeddings_array, metadata)
        logger.info(f"Ingested {len(chunks)} chunks")
        return len(chunks)

    def _metrics_timer(self, component: str, operation: str):
        from voice_optimized_rag.utils.metrics import Timer

        return Timer(self._metrics, component, operation)

    def save_index(self, path: Path | None = None) -> None:
        """Save the FAISS index to disk (no-op for Qdrant, which auto-persists)."""
        if hasattr(self._vector_store, "save"):
            self._vector_store.save(path or self._config.faiss_index_path)

    @property
    def metrics(self) -> MetricsCollector:
        """Access the metrics collector."""
        return self._metrics

    @property
    def cache(self) -> SemanticCache:
        """Access the semantic cache."""
        return self._cache

    @property
    def stream(self) -> ConversationStream:
        """Access the conversation stream."""
        return self._stream
