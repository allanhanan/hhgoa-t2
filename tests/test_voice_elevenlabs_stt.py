from __future__ import annotations

import pytest

from examples.voice_demo import handle_utterance
from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.voice.stt import ElevenLabsSTT, create_stt
from voice_optimized_rag.voice.tts import create_tts


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    def __init__(self) -> None:
        self.last_request = None

    async def post(self, url, headers=None, data=None, files=None):
        self.last_request = {
            "url": url,
            "headers": headers,
            "data": data,
            "files": files,
        }
        return FakeResponse({"text": "what was the immediate impact of the Manhattan Project?"})


def test_voice_defaults_use_elevenlabs_stt_and_disable_tts() -> None:
    config = VORConfig()

    assert config.stt_provider == "elevenlabs"
    assert config.tts_provider == "none"
    assert config.elevenlabs_stt_model == "scribe_v2"
    assert create_tts(config.tts_provider) is None


@pytest.mark.asyncio
async def test_elevenlabs_stt_posts_wav_to_speech_to_text() -> None:
    stt = create_stt("elevenlabs", api_key="test-key", model="scribe_v2")
    assert isinstance(stt, ElevenLabsSTT)

    fake_client = FakeAsyncClient()
    stt._client = fake_client

    text = await stt.transcribe(b"\x00\x00" * 1600, sample_rate=16000)

    assert text == "what was the immediate impact of the Manhattan Project?"
    assert fake_client.last_request["url"] == "https://api.elevenlabs.io/v1/speech-to-text"
    assert fake_client.last_request["headers"] == {"xi-api-key": "test-key"}
    assert fake_client.last_request["data"] == {"model_id": "scribe_v2"}
    filename, file_obj, content_type = fake_client.last_request["files"]["file"]
    assert filename == "utterance.wav"
    assert content_type == "audio/wav"
    assert file_obj.read(4) == b"RIFF"


def test_elevenlabs_stt_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ElevenLabs STT requires"):
        create_stt("elevenlabs")


class FakeSTT:
    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        assert audio_data
        assert sample_rate == 16000
        return "what was the immediate impact of the Manhattan Project?"


class FakeRouter:
    class Metrics:
        cache_hit_rate = 0.0

    metrics = Metrics()

    async def query_stream(self, text: str):
        assert text == "what was the immediate impact of the Manhattan Project?"
        yield "The retrieved context says the success obliterated "
        yield "hundreds of thousands of innocent lives."


@pytest.mark.asyncio
async def test_voice_utterance_flow_returns_text_answer_without_tts() -> None:
    text, response, timings = await handle_utterance(
        utterance=b"\x00\x01" * 1600,
        sample_rate=16000,
        stt=FakeSTT(),
        router=FakeRouter(),
        tts=None,
    )

    assert text == "what was the immediate impact of the Manhattan Project?"
    assert "retrieved context" in response
    assert timings["tts_ms"] == 0.0
