"""Ollama LLM provider implementation for local models."""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from voice_optimized_rag.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    """LLM provider using a local Ollama server."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._client = httpx.AsyncClient(timeout=60.0)

    async def generate(self, prompt: str, context: str = "") -> str:
        messages = self._build_messages(prompt, context)
        response = await self._client.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self._temperature},
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def stream(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        messages = self._build_messages(prompt, context)
        async with self._client.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": self._temperature},
            },
        ) as response:
            response.raise_for_status()
            import json
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
