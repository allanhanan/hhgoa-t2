"""Circuit breaker for per-provider failure tracking and automatic fallback."""
from __future__ import annotations

import time
from enum import Enum


class State(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Provider is down, reject calls
    HALF_OPEN = "half_open" # Testing recovery


class CircuitBreaker:
    """Simple circuit breaker for LLM provider failover.

    - CLOSED: normal operation, track failures
    - OPEN: after `failure_threshold` consecutive failures, reject all calls
    - HALF_OPEN: after `recovery_timeout` seconds, allow one test call
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = State.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0

    @property
    def state(self) -> State:
        if self._state == State.OPEN:
            # Check if recovery timeout has elapsed
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = State.HALF_OPEN
        return self._state

    def is_available(self) -> bool:
        """Check if the provider should be attempted."""
        s = self.state
        return s in (State.CLOSED, State.HALF_OPEN)

    def record_success(self) -> None:
        """Record a successful call — reset failure count."""
        self._failure_count = 0
        self._state = State.CLOSED

    def record_failure(self) -> None:
        """Record a failed call — increment count, possibly open circuit."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = State.OPEN

    def __repr__(self) -> str:
        return f"CircuitBreaker({self.name!r}, state={self.state.value}, failures={self._failure_count})"
