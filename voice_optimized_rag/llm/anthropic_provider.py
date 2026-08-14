"""Anthropic LLM provider implementation."""

from __future__ import annotations

from typing import AsyncIterator

from voice_optimized_rag.llm.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """LLM provider using Anthropic's API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        temperature: float = 0.3,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install voice-optimized-rag[anthropic]")

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def _build_anthropic_params(self, prompt: str, context: str) -> dict:
        system = ""
        if context:
            system = (
                "Use the following context to answer the user's question. "
                "If the context is not relevant, answer from your own knowledge.\n\n"
                f"Context:\n{context}"
            )
        params: dict = {
            "model": self._model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
        }
        if system:
            params["system"] = system
        return params

    async def generate(self, prompt: str, context: str = "") -> str:
        response = await self._client.messages.create(
            **self._build_anthropic_params(prompt, context)
        )
        return response.content[0].text

    async def stream(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        async with self._client.messages.stream(
            **self._build_anthropic_params(prompt, context)
        ) as stream:
            async for text in stream.text_stream:
                yield text
