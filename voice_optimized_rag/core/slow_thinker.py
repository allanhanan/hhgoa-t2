"""Slow Thinker: Background agent that listens, predicts, and pre-fetches.

This agent runs as a background asyncio task. It subscribes to the conversation
stream, uses an LLM to predict what topics/questions the user might ask next,
retrieves relevant documents from the vector store, and pre-populates the
semantic cache so the Fast Talker can respond instantly.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.conversation_stream import (
    ConversationStream,
    EventType,
    StreamEvent,
)
from voice_optimized_rag.core.semantic_cache import SemanticCache
from voice_optimized_rag.llm.base import LLMProvider
from voice_optimized_rag.retrieval.embeddings import EmbeddingProvider
from voice_optimized_rag.retrieval.vector_store import FAISSVectorStore
from voice_optimized_rag.utils.logging import get_logger
from voice_optimized_rag.utils.metrics import MetricsCollector, Timer

logger = get_logger("slow_thinker")

PREDICTION_PROMPT = """Based on the conversation so far, predict the {n} most likely topics the user will ask about next. For each topic, write a short document-style description (not a question) that would match relevant knowledge base content.

Conversation:
{conversation}

Latest user message: {latest}

Return ONLY a numbered list, one topic per line. Write each as a descriptive phrase that matches documentation content. Example format:
1. Enterprise plan pricing features unlimited contacts dedicated support custom SLA
2. API authentication using Bearer token API key generation permissions
3. Slack integration deal notifications slash commands setup"""

KEYWORD_EXTRACTION_PROMPT = """Extract the key topics and entities from this text. Return only a comma-separated list of keywords.

Text: {text}

Keywords:"""


class SlowThinker:
    """Background agent that continuously pre-fetches context into the cache.

    Architecture:
        1. Subscribes to conversation stream events
        2. On UserUtterance: predicts follow-up topics, retrieves, caches
        3. On SilenceDetected: prefetches broader context
        4. On TopicShift: clears stale cache, prefetches for new topic
        5. On PriorityRetrieval: immediate retrieval for cache miss fallback
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
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_prediction_time: float = 0
        self._last_partial_time: float = 0

    async def start(self) -> None:
        """Start the background processing loop."""
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Slow Thinker started")

    async def stop(self) -> None:
        """Stop the background processing loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Slow Thinker stopped")

    async def _run(self) -> None:
        """Main processing loop — subscribe to stream and handle events."""
        subscription = self._stream.subscribe()
        try:
            async for event in subscription:
                if not self._running:
                    break
                try:
                    await self._handle_event(event)
                except Exception as e:
                    logger.error(f"Error handling event: {e}")
        except asyncio.CancelledError:
            pass
        finally:
            if hasattr(subscription, "unsubscribe"):
                await subscription.unsubscribe()

    async def _handle_event(self, event: StreamEvent) -> None:
        """Route events to appropriate handlers."""
        if event.event_type == EventType.USER_UTTERANCE:
            await self._on_user_utterance(event)
        elif event.event_type == EventType.PARTIAL_UTTERANCE:
            await self._on_partial_utterance(event)
        elif event.event_type == EventType.SILENCE_DETECTED:
            await self._on_silence(event)
        elif event.event_type == EventType.TOPIC_SHIFT:
            await self._on_topic_shift(event)
        elif event.event_type == EventType.PRIORITY_RETRIEVAL:
            await self._on_priority_retrieval(event)

    async def _on_user_utterance(self, event: StreamEvent) -> None:
        """Handle a new user utterance: predict, retrieve, and cache."""
        # Rate limiting
        now = time.time()
        if now - self._last_prediction_time < self._config.slow_thinker_rate_limit:
            return
        self._last_prediction_time = now

        # Step 1: Retrieve context for the current utterance
        # Skip if we just did a partial retrieval a fraction of a second ago
        if now - self._last_partial_time > 0.5:
            await self._retrieve_and_cache(event.text)

        # Step 2: Predict follow-up topics and pre-fetch
        predictions = await self._predict_followups(event.text)
        self._metrics.increment("predictions_made", len(predictions))

        # Step 3: Retrieve for each prediction (in parallel)
        if predictions:
            tasks = [self._retrieve_and_cache(pred) for pred in predictions]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _on_partial_utterance(self, event: StreamEvent) -> None:
        """Handle streaming partial text by speculatively retrieving mid-sentence."""
        words = event.text.split()
        if len(words) < 4:
            return # Too short to be semantically useful
            
        now = time.time()
        # Debounce: wait at least 0.5 seconds between partial retrievals to avoid spam
        if now - self._last_partial_time < 0.5:
            return
            
        self._last_partial_time = now
        with Timer(self._metrics, "slow_thinker", "speculative_retrieval"):
            await self._retrieve_and_cache(event.text, top_k=self._config.prefetch_top_k)

    async def _on_silence(self, event: StreamEvent) -> None:
        """On silence, prefetch broader context around the current topic."""
        conversation = self._stream.get_conversation_text(max_turns=4)
        if not conversation:
            return

        # Extract key topics from recent conversation
        keywords = await self._extract_keywords(conversation)
        if keywords:
            await self._retrieve_and_cache(keywords)

    async def _on_topic_shift(self, event: StreamEvent) -> None:
        """On topic shift, clear stale cache and prefetch for new topic."""
        # Clear entries older than half the TTL
        await self._cache.clear_stale(self._config.cache_ttl_seconds / 2)
        # Prefetch for the new topic
        if event.text:
            await self._retrieve_and_cache(event.text)

    async def _on_priority_retrieval(self, event: StreamEvent) -> None:
        """Immediate retrieval triggered by a Fast Talker cache miss."""
        with Timer(self._metrics, "slow_thinker", "priority_retrieval"):
            await self._retrieve_and_cache(event.text, top_k=self._config.prefetch_top_k * 2)

    async def _predict_followups(self, latest_utterance: str) -> list[str]:
        """Use the LLM to predict likely follow-up questions."""
        if self._config.prediction_strategy == "keyword":
            return await self._predict_keyword(latest_utterance)
        return await self._predict_llm(latest_utterance)

    async def _predict_llm(self, latest_utterance: str) -> list[str]:
        """LLM-based prediction of follow-up topics."""
        with Timer(self._metrics, "slow_thinker", "prediction"):
            conversation = self._stream.get_conversation_text(max_turns=6)
            prompt = PREDICTION_PROMPT.format(
                n=self._config.max_predictions,
                conversation=conversation,
                latest=latest_utterance,
            )
            try:
                response = await self._llm.generate(prompt)
                predictions = []
                for line in response.strip().split("\n"):
                    line = line.strip()
                    # Strip numbering like "1. " or "- "
                    if line and line[0].isdigit():
                        line = line.split(".", 1)[-1].strip()
                    elif line.startswith("- "):
                        line = line[2:].strip()
                    if line:
                        predictions.append(line)
                return predictions[: self._config.max_predictions]
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                return await self._predict_keyword(latest_utterance)

    async def _predict_keyword(self, text: str) -> list[str]:
        """Simple keyword-based prediction (fallback)."""
        keywords = await self._extract_keywords(text)
        return [keywords] if keywords else []

    async def _extract_keywords(self, text: str) -> str:
        """Extract keywords from text using the LLM."""
        try:
            prompt = KEYWORD_EXTRACTION_PROMPT.format(text=text)
            return await self._llm.generate(prompt)
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            # Simple fallback: return the text itself as a query
            return text

    async def _retrieve_and_cache(self, query: str, top_k: int | None = None) -> None:
        """Retrieve documents for a query and store in semantic cache.

        Each document chunk is cached with its OWN embedding (not the query
        embedding). This ensures that when the Fast Talker searches the cache
        with the user's actual question, it finds the most relevant chunks
        regardless of which prediction query originally retrieved them.
        """
        k = top_k or self._config.prefetch_top_k

        with Timer(self._metrics, "slow_thinker", "retrieval"):
            # Embed the query
            query_embedding = await self._embeddings.embed_single(query)

            # Search vector store — include document embeddings for cache keys
            results = self._vector_store.search(
                query_embedding, top_k=k, include_embeddings=True,
            )

            if not results:
                return

            # Cache each result with its DOCUMENT embedding as the key
            for result in results:
                cache_key = result.embedding if result.embedding is not None else query_embedding
                await self._cache.put(
                    query_embedding=cache_key,
                    text=result.text,
                    metadata=result.metadata,
                    relevance_score=result.score,
                    ttl=self._config.cache_ttl_seconds,
                )

        self._metrics.increment("prefetch_operations")
        logger.debug(f"Pre-fetched {len(results)} chunks for: {query[:50]}...")
