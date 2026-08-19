"""Local LLM client supporting both LM Studio (OpenAI API on port 1234) and llama.cpp (port 8081)."""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

from app.config import (
    LMSTUDIO_URL,
    LLAMA_CPP_URL,
    LOCAL_LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_REPEAT_PENALTY,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


def _build_prompt(query: str, passages: list[str]) -> str:
    """Build the ChatML-format prompt for SmolLM2 / Qwen2.5."""
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nContext:\n{context}\n\n"
        f"Question: {query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


_shared_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Get or create shared persistent HTTP client with connection pooling."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        fast_timeout = httpx.Timeout(connect=0.15, read=5.0, write=2.0, pool=2.0)
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
        _shared_client = httpx.AsyncClient(timeout=fast_timeout, limits=limits)
    return _shared_client


async def generate_stream(
    query: str,
    passages: list[str],
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    """Stream tokens from LM Studio (port 2000 / 1234) or llama.cpp (port 8081).

    Yields individual tokens as they arrive via SSE.
    """
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    client = _get_client()

    # 1. Try llama.cpp server (http://127.0.0.1:8081/completion)
    prompt = _build_prompt(query, passages)
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens or LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "repeat_penalty": LLM_REPEAT_PENALTY,
        "stop": ["<|im_end|>", "<|im_start|>"],
        "stream": True,
    }

    try:
        async with client.stream(
            "POST",
            f"{LLAMA_CPP_URL.rstrip('/')}/completion",
            json=payload,
        ) as response:
            if response.status_code == 200:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            token = chunk.get("content", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue
                return
    except Exception as e:
        logger.debug(f"llama.cpp stream skipped: {e}")

    # 2. Fallback to LM Studio OpenAI API (http://127.0.0.1:2000/v1/chat/completions)
    try:
        lm_payload = {
            "model": LOCAL_LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            "max_tokens": max_tokens or LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "stream": True,
        }

        async with client.stream(
            "POST",
            f"{LMSTUDIO_URL.rstrip('/')}/chat/completions",
            json=lm_payload,
        ) as response:
            if response.status_code == 200:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            token = chunk["choices"][0]["delta"].get("content", "")
                            if token:
                                yield token
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                return
    except Exception as e:
        logger.debug(f"LM Studio stream skipped: {e}")


async def generate(query: str, passages: list[str]) -> str:
    """Non-streaming generation — collect all tokens."""
    tokens = []
    async for token in generate_stream(query, passages):
        tokens.append(token)
    return "".join(tokens)


async def health_check() -> bool:
    """Check if llama.cpp or LM Studio server is reachable."""
    # Check llama.cpp first
    try:
        async with httpx.AsyncClient(timeout=0.5) as client:
            resp = await client.get(f"{LLAMA_CPP_URL.rstrip('/')}/health")
            if resp.status_code == 200:
                return True
    except Exception:
        pass

    # Check LM Studio fallback
    try:
        async with httpx.AsyncClient(timeout=0.5) as client:
            resp = await client.get(f"{LMSTUDIO_URL.rstrip('/')}/models")
            if resp.status_code == 200:
                return True
    except Exception:
        pass

    return False

