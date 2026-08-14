"""Abstract LLM provider interface and factory."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import AsyncIterator

from voice_optimized_rag.config import VORConfig


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, context: str = "") -> str:
        """Generate a complete response.

        Args:
            prompt: The user prompt / instruction.
            context: Optional retrieved context to include.

        Returns:
            The full generated text.
        """

    @abstractmethod
    async def stream(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        """Stream a response token by token.

        Args:
            prompt: The user prompt / instruction.
            context: Optional retrieved context to include.

        Yields:
            Text chunks as they are generated.
        """

    def _build_messages(self, prompt: str, context: str) -> list[dict[str, str]]:
        """Build a chat messages list from prompt and context."""
        messages: list[dict[str, str]] = []
        if context:
            system_prompt = (
                "You are a helpful AI assistant.\n"
                "Rules:\n"
                "1. Base your answer strictly on the provided context.\n"
                "2. Do NOT use outside knowledge.\n"
                "3. The provided context has already been filtered for relevance. You must summarize the context to answer the user's query to the best of your ability.\n\n"
                "Retrieved Context:\n"
                f"{context}"
            )
            messages.append({
                "role": "system",
                "content": system_prompt,
            })
        messages.append({"role": "user", "content": prompt})
        return messages


def create_llm(config: VORConfig) -> LLMProvider:
    """Factory function to create an LLM provider from config."""
    provider = config.llm_provider

    if provider == "openai":
        from voice_optimized_rag.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(
            api_key=config.llm_api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
            base_url=config.llm_base_url,
            max_tokens=config.llm_max_tokens,
        )
    elif provider == "anthropic":
        from voice_optimized_rag.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            api_key=config.llm_api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
        )
    elif provider == "ollama":
        from voice_optimized_rag.llm.ollama_provider import OllamaProvider
        return OllamaProvider(
            model=config.llm_model,
            base_url=config.llm_base_url or "http://localhost:11434",
            temperature=config.llm_temperature,
        )
    elif provider == "gemini":
        from voice_optimized_rag.llm.gemini_provider import GeminiProvider
        # Use gemini_api_key if set, otherwise fall back to llm_api_key
        api_key = config.gemini_api_key or config.llm_api_key
        return GeminiProvider(
            api_key=api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
            vertex_project=config.vertex_project,
            vertex_location=config.vertex_location,
        )
    elif provider == "groq":
        from voice_optimized_rag.llm.openai_provider import OpenAIProvider
        key = (config.llm_api_key or os.environ.get("GROQ_API_KEY", "") or os.environ.get("VOR_LLM_API_KEY", "")).strip()
        return OpenAIProvider(
            api_key=key,
            model=config.llm_model,
            temperature=config.llm_temperature,
            base_url=config.llm_base_url or "https://api.groq.com/openai/v1",
            max_tokens=config.llm_max_tokens,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
