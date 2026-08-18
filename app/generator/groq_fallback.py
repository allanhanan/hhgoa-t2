"""Groq API fallback generator for when local llama.cpp is unavailable."""
from __future__ import annotations

from typing import AsyncIterator

from app.config import GROQ_API_KEY, GROQ_MODEL, LLM_MAX_TOKENS, SYSTEM_PROMPT


async def generate_stream(
    query: str,
    passages: list[str],
) -> AsyncIterator[str]:
    """Stream tokens from Groq API as fallback."""
    if not GROQ_API_KEY:
        yield "Error: Groq API key not configured."
        return

    from groq import AsyncGroq

    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))

    client = AsyncGroq(api_key=GROQ_API_KEY)
    stream = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.1,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


async def health_check() -> bool:
    """Check if Groq API is reachable."""
    return bool(GROQ_API_KEY)
