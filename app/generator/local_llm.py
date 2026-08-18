"""Local LLM client via llama.cpp HTTP server (streaming)."""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from app.config import (
    LLAMA_CPP_URL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_REPEAT_PENALTY,
    SYSTEM_PROMPT,
)


def _build_prompt(query: str, passages: list[str]) -> str:
    """Build the ChatML-format prompt for SmolLM2-135M."""
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
    """Stream tokens from the local llama.cpp server.

    Yields individual tokens as they arrive via SSE.
    """
    prompt = _build_prompt(query, passages)
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens or LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "repeat_penalty": LLM_REPEAT_PENALTY,
        "stop": ["<|im_end|>", "<|im_start|>"],
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST",
            f"{LLAMA_CPP_URL}/completion",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        token = chunk.get("content", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue


async def generate(query: str, passages: list[str]) -> str:
    """Non-streaming generation — collect all tokens."""
    tokens = []
    async for token in generate_stream(query, passages):
        tokens.append(token)
    return "".join(tokens)


async def health_check() -> bool:
    """Check if the llama.cpp server is reachable."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{LLAMA_CPP_URL}/health")
            return resp.status_code == 200
    except Exception:
        return False
