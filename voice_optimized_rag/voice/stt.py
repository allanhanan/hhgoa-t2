"""Speech-to-Text abstraction layer."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("stt")


class STTProvider(ABC):
    """Abstract base class for speech-to-text providers."""

    @abstractmethod
    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes (16-bit PCM).
            sample_rate: Audio sample rate in Hz.

        Returns:
            Transcribed text.
        """


class WhisperSTT(STTProvider):
    """Local STT using faster-whisper (no API needed)."""

    def __init__(self, model_size: str = "base.en") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("Install faster-whisper: pip install voice-optimized-rag[voice]")
        self._model = WhisperModel(model_size, compute_type="int8")
        logger.info(f"Loaded Whisper model: {model_size}")

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        import asyncio
        import io
        import numpy as np

        # Convert bytes to float32 numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(audio_np, language="en"),
        )

        # Collect all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        return " ".join(text_parts)


class OpenAISTT(STTProvider):
    """STT using OpenAI's Whisper API."""

    def __init__(self, api_key: str) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Install openai: pip install voice-optimized-rag[openai]")
        self._client = AsyncOpenAI(api_key=api_key)

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        import io
        import wave

        # Wrap raw PCM in a WAV container for the API
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        wav_buffer.seek(0)
        wav_buffer.name = "audio.wav"

        response = await self._client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_buffer,
        )
        return response.text


class ElevenLabsSTT(STTProvider):
    """STT using ElevenLabs Scribe."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "scribe_v2",
        base_url: str = "https://api.elevenlabs.io",
    ) -> None:
        import httpx

        key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        if not key:
            raise ValueError("ElevenLabs STT requires ELEVENLABS_API_KEY or VOR_ELEVENLABS_API_KEY")

        self._client = httpx.AsyncClient(timeout=120.0)
        self._api_key = key
        self._model = model
        self._base_url = base_url.rstrip("/")

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        import io
        import wave

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        wav_buffer.seek(0)

        files = {
            "file": ("utterance.wav", wav_buffer, "audio/wav"),
        }
        data = {
            "model_id": self._model,
        }
        response = await self._client.post(
            f"{self._base_url}/v1/speech-to-text",
            headers={"xi-api-key": self._api_key},
            data=data,
            files=files,
        )
        status_code = getattr(response, "status_code", 200)
        if status_code >= 400:
            try:
                err_json = response.json()
                msg = err_json.get("detail", {}).get("message", response.text)
            except Exception:
                msg = getattr(response, "text", str(response))
            logger.error(f"ElevenLabs STT failed (HTTP {status_code}): {msg}")
            raise RuntimeError(f"ElevenLabs STT API error ({status_code}): {msg}")

        if hasattr(response, "raise_for_status"):
            response.raise_for_status()

        payload = response.json()
        return str(payload.get("text", "")).strip()


def create_stt(provider: str, **kwargs) -> STTProvider:
    """Factory function to create an STT provider."""
    if provider == "elevenlabs":
        return ElevenLabsSTT(
            api_key=kwargs.get("api_key", ""),
            model=kwargs.get("model", "scribe_v2"),
            base_url=kwargs.get("base_url", "https://api.elevenlabs.io"),
        )
    elif provider == "whisper":
        return WhisperSTT(model_size=kwargs.get("model_size", "base.en"))
    elif provider == "openai":
        return OpenAISTT(api_key=kwargs["api_key"])
    else:
        raise ValueError(f"Unknown STT provider: {provider}")
