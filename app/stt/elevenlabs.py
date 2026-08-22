"""ElevenLabs Scribe v2 Realtime STT client (WebSocket streaming)."""
from __future__ import annotations

import asyncio
import json
import base64
from typing import AsyncIterator

import websockets

from app.config import ELEVENLABS_API_KEY

SCRIBE_WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"


async def stream_transcribe(
    audio_chunks: AsyncIterator[bytes],
    sample_rate: int = 16000,
    encoding: str = "pcm_s16le",
) -> AsyncIterator[dict]:
    """Stream audio chunks to ElevenLabs Scribe v2 Realtime and yield transcripts.

    Args:
        audio_chunks: Async iterator of PCM audio bytes (100ms chunks recommended).
        sample_rate: Audio sample rate in Hz.
        encoding: Audio encoding format.

    Yields:
        Dict with keys: 'text' (str), 'is_final' (bool), 'confidence' (float).
    """
    if not ELEVENLABS_API_KEY:
        yield {"text": "", "is_final": True, "confidence": 0.0, "error": "No API key"}
        return

    url = f"{SCRIBE_WS_URL}?model_id=scribe_v2&sample_rate={sample_rate}&encoding={encoding}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}

    try:
        async with websockets.connect(url, extra_headers=headers) as ws:
            # Send audio chunks in background
            async def send_audio():
                async for chunk in audio_chunks:
                    msg = json.dumps({
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    })
                    await ws.send(msg)
                # Signal end of audio
                await ws.send(json.dumps({"audio": "", "eof": True}))

            send_task = asyncio.create_task(send_audio())

            # Receive transcripts
            try:
                async for message in ws:
                    data = json.loads(message)
                    msg_type = data.get("type", "")

                    if msg_type == "transcript":
                        yield {
                            "text": data.get("text", ""),
                            "is_final": data.get("is_final", False),
                            "confidence": data.get("confidence", 0.0),
                        }

                    if msg_type == "error":
                        yield {
                            "text": "",
                            "is_final": True,
                            "confidence": 0.0,
                            "error": data.get("message", "Unknown STT error"),
                        }
                        break
            finally:
                send_task.cancel()

    except Exception as e:
        yield {
            "text": "",
            "is_final": True,
            "confidence": 0.0,
            "error": f"WebSocket error: {e}",
        }


async def transcribe_audio_rest(audio_bytes: bytes, content_type: str = "audio/webm") -> dict[str, str]:
    """Transcribe audio file bytes using ElevenLabs Speech-to-Text REST API.

    Returns:
        Dict with key 'text' (str) or 'error' (str).
    """
    if not ELEVENLABS_API_KEY:
        return {"text": "", "error": "ELEVENLABS_API_KEY is not set in .env"}

    import httpx

    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}

    files = {
        "file": ("audio.webm", audio_bytes, content_type),
        "model_id": (None, "scribe_v1"),
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, headers=headers, files=files)
            if resp.status_code == 200:
                res_data = resp.json()
                return {"text": res_data.get("text", "")}
            return {"text": "", "error": f"ElevenLabs STT API Error: {resp.status_code} - {resp.text}"}
    except Exception as e:
        return {"text": "", "error": f"STT request failed: {e}"}
