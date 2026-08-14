"""Google Gemini LLM provider implementation.

Supports both standard Gemini API (via API key) and Vertex AI (via GCP auth).
When VERTEX_PROJECT is set, uses Vertex AI backend. Otherwise uses the standard
Gemini API with the API key.
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

from voice_optimized_rag.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    """LLM provider using Google Gemini.

    Authentication (in order of priority):
      1. Standard: api_key or GEMINI_API_KEY / GOOGLE_API_KEY
      2. Vertex AI: vertex_project or VERTEX_PROJECT + gcloud auth
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        vertex_project: str = "",
        vertex_location: str = "us-central1",
    ) -> None:
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "Install google-genai: pip install google-genai"
            )

        self._genai = genai
        self._model = model
        self._temperature = temperature

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        project = vertex_project or os.environ.get("VERTEX_PROJECT", "")
        location = vertex_location or os.environ.get("VERTEX_LOCATION", "us-central1")

        if project:
            # Vertex AI mode (GCP authentication via gcloud)
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
        elif key:
            # Standard Gemini API key mode
            self._client = genai.Client(api_key=key)
        else:
            raise ValueError(
                "Gemini requires either GEMINI_API_KEY (standard) or "
                "VERTEX_PROJECT (Vertex AI). Set one in your environment."
            )

    def _build_gemini_config(self):
        """Build Gemini generation config."""
        from google.genai import types
        return types.GenerateContentConfig(
            max_output_tokens=1024,
            temperature=self._temperature,
        )

    async def generate(self, prompt: str, context: str = "") -> str:
        full_prompt = self._format_prompt(prompt, context)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=[full_prompt],
                config=self._build_gemini_config(),
            ),
        )
        return response.text.strip() if response.text else ""

    async def stream(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        full_prompt = self._format_prompt(prompt, context)

        loop = asyncio.get_event_loop()
        # Gemini's streaming is synchronous in google-genai, so run in executor
        # and yield chunks as they come. For simplicity, generate full response
        # and yield it, since google-genai doesn't have async streaming yet.
        response = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=[full_prompt],
                config=self._build_gemini_config(),
            ),
        )
        text = response.text.strip() if response.text else ""
        # Yield in word-sized chunks to simulate streaming
        words = text.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

    def _format_prompt(self, prompt: str, context: str) -> str:
        """Format prompt with optional context for Gemini (no system message support)."""
        if context:
            return (
                "Use the following context to answer the user's question. "
                "If the context is not relevant, answer from your own knowledge.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {prompt}"
            )
        return prompt
