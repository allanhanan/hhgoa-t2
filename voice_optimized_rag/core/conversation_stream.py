"""Async conversation event stream.

Provides an event bus that the Slow Thinker subscribes to and the Memory Router
publishes to. Maintains a sliding window of recent conversation history.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


class EventType(Enum):
    USER_UTTERANCE = "user_utterance"       # Complete user speech
    PARTIAL_UTTERANCE = "partial_utterance" # In-progress user speech
    AGENT_RESPONSE = "agent_response"       # Complete agent response
    AGENT_CHUNK = "agent_chunk"             # Streaming response chunk
    SILENCE_DETECTED = "silence_detected"   # Extended silence
    TOPIC_SHIFT = "topic_shift"             # Explicit context clear
    PRIORITY_RETRIEVAL = "priority_retrieval" # Fast talker needs help


@dataclass
class StreamEvent:
    """A single event in the conversation stream."""
    event_type: EventType
    text: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ConversationStream:
    """Async event bus for conversation events with sliding-window history.

    The Memory Router publishes events here. The Slow Thinker subscribes
    to receive them and trigger prefetching logic.
    """

    def __init__(self, window_size: int = 10) -> None:
        self._subscribers: list[asyncio.Queue[StreamEvent]] = []
        self._history: deque[StreamEvent] = deque(maxlen=window_size)
        self._window_size = window_size

    async def publish(self, event: StreamEvent) -> None:
        """Publish an event to all subscribers and add to history."""
        self._history.append(event)
        for queue in self._subscribers:
            await queue.put(event)

    def subscribe(self) -> AsyncIterator[StreamEvent]:
        """Create a new subscription that yields events as they arrive."""
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        return _SubscriptionIterator(queue, self._subscribers)

    @property
    def history(self) -> list[StreamEvent]:
        """Get the recent conversation history."""
        return list(self._history)

    def get_conversation_text(self, max_turns: int | None = None) -> str:
        """Get recent conversation as formatted text.

        Args:
            max_turns: Max number of turns to include (None = all in window).

        Returns:
            Formatted conversation string.
        """
        events = list(self._history)
        if max_turns:
            events = events[-max_turns:]

        lines: list[str] = []
        for event in events:
            if event.event_type == EventType.USER_UTTERANCE:
                lines.append(f"User: {event.text}")
            elif event.event_type == EventType.AGENT_RESPONSE:
                lines.append(f"Assistant: {event.text}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear conversation history."""
        self._history.clear()


class _SubscriptionIterator:
    """Async iterator for a subscription queue."""

    def __init__(
        self,
        queue: asyncio.Queue[StreamEvent],
        subscribers: list[asyncio.Queue[StreamEvent]],
    ) -> None:
        self._queue = queue
        self._subscribers = subscribers

    def __aiter__(self) -> _SubscriptionIterator:
        return self

    async def __anext__(self) -> StreamEvent:
        try:
            return await self._queue.get()
        except asyncio.CancelledError:
            # Clean up subscription on cancellation
            if self._queue in self._subscribers:
                self._subscribers.remove(self._queue)
            raise

    async def unsubscribe(self) -> None:
        """Remove this subscription."""
        if self._queue in self._subscribers:
            self._subscribers.remove(self._queue)
