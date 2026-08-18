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


async def generate_stream(
    query: str,
    passages: list[str],
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    """Stream tokens from LM Studio (port 2000 / 1234) or llama.cpp (port 8081).

    Yields individual tokens as they arrive via SSE.
    """
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))

    # 1. Try LM Studio OpenAI API (http://127.0.0.1:2000/v1/chat/completions)
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

        async with httpx.AsyncClient(timeout=2.0) as client:
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

    # 2. Fallback to llama.cpp (http://localhost:8081/completion)
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
        async with httpx.AsyncClient(timeout=2.0) as client:
            async with client.stream(
                "POST",
                f"{LLAMA_CPP_URL.rstrip('/')}/completion",
                json=payload,
            ) as response:
                response.raise_for_status()
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
    except Exception as e:
        logger.debug(f"llama.cpp stream skipped: {e}")


async def generate(query: str, passages: list[str]) -> str:
    """Non-streaming generation — collect all tokens."""
    tokens = []
    async for token in generate_stream(query, passages):
        tokens.append(token)
    return "".join(tokens)


async def health_check() -> bool:
    """Check if LM Studio or llama.cpp server is reachable."""
    # Check LM Studio
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(f"{LMSTUDIO_URL.rstrip('/')}/models")
            if resp.status_code == 200:
                return True
    except Exception:
        pass

    # Check llama.cpp
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(f"{LLAMA_CPP_URL.rstrip('/')}/health")
            if resp.status_code == 200:
                return True
    except Exception:
        pass

    return False
