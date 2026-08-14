"""Text-to-Speech abstraction layer."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("tts")


class TTSProvider(ABC):
    """Abstract base class for text-to-speech providers."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio.

        Args:
            text: Text to convert to speech.

        Returns:
            Raw audio bytes (16-bit PCM).
        """


class EdgeTTS(TTSProvider):
    """Free TTS using Microsoft Edge's TTS service (no API key needed)."""

    def __init__(self, voice: str = "en-US-AriaNeural") -> None:
        try:
            import edge_tts
        except ImportError:
            raise ImportError("Install edge-tts: pip install voice-optimized-rag[voice]")
        self._voice = voice

    async def synthesize(self, text: str) -> bytes:
        import edge_tts
        import io

        communicate = edge_tts.Communicate(text, self._voice)
        audio_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)


class OpenAITTS(TTSProvider):
    """TTS using OpenAI's API."""

    def __init__(self, api_key: str, voice: str = "alloy", model: str = "tts-1") -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Install openai: pip install voice-optimized-rag[openai]")
        self._client = AsyncOpenAI(api_key=api_key)
        self._voice = voice
        self._model = model

    async def synthesize(self, text: str) -> bytes:
        response = await self._client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format="pcm",
        )
        return response.content


class ElevenLabsTTS(TTSProvider):
    """TTS using ElevenLabs, returning 16 kHz PCM for local playback."""

    def __init__(
        self,
        api_key: str = "",
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model: str = "eleven_multilingual_v2",
    ) -> None:
        import httpx

        key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        if not key:
            raise ValueError("ElevenLabs TTS requires ELEVENLABS_API_KEY or VOR_ELEVENLABS_API_KEY")

        self._client = httpx.AsyncClient(timeout=60.0)
        self._api_key = key
        self._voice_id = voice_id
        self._model = model

    async def synthesize(self, text: str) -> bytes:
        response = await self._client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}",
            params={"output_format": "pcm_16000"},
            headers={
                "xi-api-key": self._api_key,
                "Accept": "audio/pcm",
            },
            json={
                "text": text,
                "model_id": self._model,
            },
        )
        response.raise_for_status()
        return response.content


def create_tts(provider: str, **kwargs) -> TTSProvider | None:
    """Factory function to create a TTS provider."""
    if provider == "none":
        return None
    elif provider == "edge":
        return EdgeTTS(voice=kwargs.get("voice", "en-US-AriaNeural"))
    elif provider == "openai":
        return OpenAITTS(
            api_key=kwargs["api_key"],
            voice=kwargs.get("voice", "alloy"),
        )
    elif provider == "elevenlabs":
        return ElevenLabsTTS(
            api_key=kwargs.get("api_key", ""),
            voice_id=kwargs.get("voice_id", "21m00Tcm4TlvDq8ikWAM"),
            model=kwargs.get("model", "eleven_multilingual_v2"),
        )
    else:
        raise ValueError(f"Unknown TTS provider: {provider}")
