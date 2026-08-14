"""Fast Talker: Foreground agent that responds from cache with minimal latency.

This agent handles user queries by reading ONLY from the semantic cache (populated
by the Slow Thinker). When the cache has relevant context, responses are near-instant.
On cache miss, it gracefully degrades to either a direct vector store lookup or
parametric-only response, while signaling the Slow Thinker to prioritize retrieval.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.conversation_stream import (
    ConversationStream,
    EventType,
    StreamEvent,
)
from voice_optimized_rag.core.guardrails import GuardrailEngine
from voice_optimized_rag.core.semantic_cache import SemanticCache
from voice_optimized_rag.llm.base import LLMProvider
from voice_optimized_rag.retrieval.embeddings import EmbeddingProvider
from voice_optimized_rag.retrieval.vector_store import FAISSVectorStore, SearchResult
from voice_optimized_rag.utils.logging import get_logger
from voice_optimized_rag.utils.metrics import MetricsCollector, Timer

logger = get_logger("fast_talker")


class FastTalker:
    """Foreground agent optimized for minimal response latency.

    Flow:
        1. Receive user query
        2. Embed query (async)
        3. Check semantic cache (sub-ms)
        4. If hit: generate response with cached context
        5. If miss: fallback to direct retrieval or parametric response
        6. Stream response back
    """

    def __init__(
        self,
        config: VORConfig,
        llm: LLMProvider,
        embedding_provider: EmbeddingProvider,
        vector_store: FAISSVectorStore,
        cache: SemanticCache,
        stream: ConversationStream,
        metrics: MetricsCollector,
    ) -> None:
        self._config = config
        self._llm = llm
        self._embeddings = embedding_provider
        self._vector_store = vector_store
        self._cache = cache
        self._stream = stream
        self._metrics = metrics
        self._guardrails = GuardrailEngine(config)

    async def respond(self, query: str) -> str:
        """Generate a complete response to the user query.

        Args:
            query: The user's question/statement.

        Returns:
            The full response text.
        """
        with Timer(self._metrics, "fast_talker", "total_response") as timer:
            results, cache_status = await self._get_context(query)
            decision = self._guardrails.evaluate(query, results, cache_status)
            if not decision.allowed:
                self._metrics.increment(f"guardrail_{decision.reason}")
                return decision.message

            # Send ONLY chunks that passed the relevance gate — never weak/unrelated passages.
            context = self._format_context([r.text for r in decision.relevant_results])

            with Timer(self._metrics, "rag", "total", cache_status):
                with Timer(self._metrics, "rag", "llm", cache_status):
                    response = await self._llm.generate(query, context=context)

            validation = self._guardrails.validate_answer(response, context)
            if not validation.allowed:
                self._metrics.increment(f"guardrail_{validation.reason}")
                return validation.message

        logger.info(f"Response generated in {timer.elapsed_ms:.1f}ms (context: {len(context)} chars)")
        return response

    async def respond_stream(self, query: str) -> AsyncIterator[str]:
        """Stream a response to the user query token by token.

        Args:
            query: The user's question/statement.

        Yields:
            Text chunks as they are generated.
        """
        with Timer(self._metrics, "fast_talker", "total_response"):
            results, cache_status = await self._get_context(query)
            decision = self._guardrails.evaluate(query, results, cache_status)
            if not decision.allowed:
                self._metrics.increment(f"guardrail_{decision.reason}")
                yield decision.message
                return

            context = self._format_context([r.text for r in decision.relevant_results])

            with Timer(self._metrics, "rag", "total", cache_status):
                first_token = True
                with Timer(self._metrics, "rag", "llm", cache_status):
                    async for chunk in self._llm.stream(query, context=context):
                        if first_token:
                            self._metrics.record_latency("rag", "time_to_first_token", 0, cache_status)
                            first_token = False
                        yield chunk

    async def inspect(self, query: str) -> dict[str, object]:
        """Retrieve and evaluate grounding for a query WITHOUT generating.

        Dev/testing helper — lets the CLI report scores, the post-filter list,
        and the grounding decision using the exact same path as a live answer.
        """
        results, cache_status = await self._get_context(query)
        decision = self._guardrails.evaluate(query, results, cache_status)
        return {
            "results": results,
            "relevant_results": decision.relevant_results,
            "allowed": decision.allowed,
            "reason": decision.reason,
            "refusal_message": decision.message,
            "cache_status": cache_status,
        }

    async def _get_context(self, query: str) -> tuple[list[SearchResult], str]:
        """Retrieve candidate context from cache (fast path) or vector store (fallback).

        Strategy:
          - Cache HIT → use cached chunks (skip vector store = saves retrieval time)
          - Cache MISS → fall back to direct vector store search (normal RAG speed)

        Returns RAW retrieval results, including low-scoring ones. The caller
        applies the relevance gate (``GuardrailEngine.evaluate``) before sending
        anything to the LLM.

        The slow thinker caches documents indexed by their OWN embeddings,
        so cache hits return chunks semantically relevant to the user's actual
        query — not just to the prediction that originally fetched them.
        """
        # Step 1: Embed the query
        cache_status = "miss"
        with Timer(self._metrics, "rag", "embedding"):
            query_embedding = await self._embeddings.embed_single(query)

        # Step 2: Try semantic cache (fast path — sub-ms)
        with Timer(self._metrics, "rag", "semantic-cache") as cache_timer:
            cached = await self._cache.get(
                query_embedding,
                top_k=self._config.fast_talker_max_context_chunks,
            )

        if cached:
            cache_status = "hit"
            logger.debug(f"Cache HIT: {len(cached)} chunks in {cache_timer.elapsed_ms:.2f}ms")
            results = [
                SearchResult(
                    text=entry.text,
                    metadata=entry.metadata,
                    score=entry.relevance_score,
                    index=i,
                    embedding=entry.embedding,
                )
                for i, entry in enumerate(cached)
            ]
            return results, cache_status

        # Step 3: Cache miss — fall back to direct vector store search
        logger.debug("Cache MISS — falling back to retrieval")

        if self._config.fast_talker_fallback_to_retrieval:
            retrieval_stage = (
                "qdrant"
                if self._config.vector_store_provider == "qdrant"
                else "retrieval"
            )
            with Timer(self._metrics, "rag", retrieval_stage, cache_status):
                results = self._vector_store.search(
                    query_embedding,
                    top_k=self._config.fast_talker_max_context_chunks,
                    include_embeddings=True,
                )
                if results:
                    # Cache ONLY results that pass the relevance floor so the cache
                    # does not accumulate unrelated junk.
                    relevant = self._guardrails.filter_relevant(results)
                    for r in relevant:
                        cache_key = r.embedding if r.embedding is not None else query_embedding
                        await self._cache.put(
                            query_embedding=cache_key,
                            text=r.text,
                            metadata=r.metadata,
                            relevance_score=r.score,
                        )
                    # Signal slow thinker to prefetch more around this topic
                    await self._stream.publish(StreamEvent(
                        event_type=EventType.PRIORITY_RETRIEVAL,
                        text=query,
                    ))
                    return results, cache_status

        # No context available
        await self._stream.publish(StreamEvent(
            event_type=EventType.PRIORITY_RETRIEVAL,
            text=query,
        ))
        return [], cache_status

    def _format_context(self, chunks: list[str]) -> str:
        """Format context chunks into a single string for the LLM."""
        if not chunks:
            return ""
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(f"[{i}] {chunk}")
        return "\n\n".join(formatted)
