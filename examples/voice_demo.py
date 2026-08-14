#!/usr/bin/env python3
"""Voice input demo: microphone to ElevenLabs STT to text RAG answers.

Flow:
    microphone/audio input -> ElevenLabs STT -> text query -> embedding
    -> Qdrant retrieval -> Groq LLM -> displayed text answer
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_optimized_rag import MemoryRouter, VORConfig
from voice_optimized_rag.voice.audio_stream import AudioStream
from voice_optimized_rag.voice.stt import create_stt
from voice_optimized_rag.voice.tts import create_tts


async def handle_utterance(
    utterance: bytes,
    sample_rate: int,
    stt,
    router,
    tts=None,
) -> tuple[str, str, dict[str, object]]:
    """Transcribe one utterance, run RAG, and optionally synthesize speech."""
    stt_name = stt.__class__.__name__
    duration_s = len(utterance) / (sample_rate * 2)
    print(f"\n[VoiceDemo] Sending utterance ({len(utterance)} bytes, ~{duration_s:.2f}s) to {stt_name}...")
    stt_start = time.perf_counter()
    try:
        text = await stt.transcribe(utterance, sample_rate)
        stt_ms = (time.perf_counter() - stt_start) * 1000
        print(f"[VoiceDemo] STT response received ({stt_ms:.0f}ms): '{text.strip()}'")
    except Exception as err:
        stt_ms = (time.perf_counter() - stt_start) * 1000
        print(f"[VoiceDemo] STT Error ({stt_ms:.0f}ms): {err}")
        return "", "", {"stt_ms": stt_ms, "rag_ms": 0.0, "tts_ms": 0.0}

    if not text.strip():
        print("[VoiceDemo] No speech text transcribed, resuming listening...\n")
        return "", "", {"stt_ms": stt_ms, "rag_ms": 0.0, "tts_ms": 0.0}

    query_start = time.perf_counter()
    response_parts: list[str] = []
    async for chunk in router.query_stream(text):
        response_parts.append(chunk)
    response = "".join(response_parts)
    query_ms = (time.perf_counter() - query_start) * 1000

    tts_ms = 0.0
    if tts:
        tts_start = time.perf_counter()
        audio_data = await tts.synthesize(response)
        tts_ms = (time.perf_counter() - tts_start) * 1000
        return text, response, {"stt_ms": stt_ms, "rag_ms": query_ms, "tts_ms": tts_ms, "audio": audio_data}

    return text, response, {"stt_ms": stt_ms, "rag_ms": query_ms, "tts_ms": tts_ms}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Voice-Optimized RAG Voice Input Demo")
    parser.add_argument("--docs", type=Path, help="Directory of documents to ingest")
    parser.add_argument("--provider", default="groq", choices=["openai", "anthropic", "ollama", "gemini", "groq"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--stt", default="elevenlabs", choices=["elevenlabs", "openai", "whisper"])
    parser.add_argument("--tts", default="none", choices=["none", "edge", "openai", "elevenlabs"])
    parser.add_argument("--elevenlabs-stt-model", default=None)
    parser.add_argument("--whisper-model", default="base.en")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    config_kwargs = {
        "llm_provider": args.provider,
        "stt_provider": args.stt,
        "tts_provider": args.tts,
        "vector_store_provider": "qdrant",
        "qdrant_collection": "msmarco_xi_test",
        "embedding_provider": "sentence-transformers",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "guardrail_min_relevance_score": 0.25,
    }
    if args.model:
        config_kwargs["llm_model"] = args.model
    if args.api_key:
        config_kwargs["llm_api_key"] = args.api_key
    if args.elevenlabs_stt_model:
        config_kwargs["elevenlabs_stt_model"] = args.elevenlabs_stt_model
    if args.provider == "ollama":
        config_kwargs.setdefault("llm_model", "llama3.2")
        config_kwargs["embedding_provider"] = "ollama"
        config_kwargs["embedding_model"] = "nomic-embed-text"
        config_kwargs["embedding_dimension"] = 768
    elif args.provider == "gemini":
        config_kwargs.setdefault("llm_model", "gemini-2.5-flash")
    elif args.provider == "groq":
        config_kwargs.setdefault("llm_model", "llama-3.3-70b-versatile")
        config_kwargs["llm_base_url"] = "https://api.groq.com/openai/v1"

    config = VORConfig(**config_kwargs)

    router = MemoryRouter(config)
    audio = AudioStream(sample_rate=config.sample_rate, vad_aggressiveness=config.vad_aggressiveness)

    if args.stt == "elevenlabs":
        stt_kwargs = {
            "api_key": config.elevenlabs_api_key,
            "model": config.elevenlabs_stt_model,
        }
    elif args.stt == "openai":
        stt_kwargs = {"api_key": config.llm_api_key}
    else:
        stt_kwargs = {"model_size": args.whisper_model}
    stt = create_stt(args.stt, **stt_kwargs)

    tts = None
    if args.tts != "none":
        tts_kwargs = {}
        if args.tts == "openai":
            tts_kwargs = {"api_key": config.llm_api_key}
        elif args.tts == "elevenlabs":
            tts_kwargs = {
                "api_key": config.elevenlabs_api_key,
                "voice_id": config.elevenlabs_voice_id,
                "model": config.elevenlabs_model,
            }
        tts = create_tts(args.tts, **tts_kwargs)

    print("=" * 60)
    print("  Voice-Optimized RAG - Voice Input Demo")
    print("  Speak into your microphone; answers are displayed as text")
    print("=" * 60)

    await router.start(log_level=args.log_level)

    if args.docs and args.docs.is_dir():
        print(f"\nIngesting documents from {args.docs}...")
        count = await router.ingest_directory(args.docs)
        print(f"Ingested {count} chunks.")
        router.save_index()

    print("\nListening... (Ctrl+C to stop)\n")

    try:
        async for utterance in audio.listen():
            text, response, timings = await handle_utterance(
                utterance=utterance,
                sample_rate=config.sample_rate,
                stt=stt,
                router=router,
                tts=tts,
            )
            if not text.strip():
                continue

            print(f"You: {text}  [STT: {timings['stt_ms']:.0f}ms]")
            print(f"Assistant: {response}")

            hit_rate = router.metrics.cache_hit_rate
            metrics = f"STT: {timings['stt_ms']:.0f}ms | RAG: {timings['rag_ms']:.0f}ms | Cache: {hit_rate:.0%}"
            if tts and "audio" in timings:
                await audio.play(timings["audio"])
                metrics += f" | TTS: {timings['tts_ms']:.0f}ms"
            print(f"  [{metrics}]\n")

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[VoiceDemo] Stopped listening (Ctrl+C). Exiting cleanly.")
    finally:
        await router.stop()
        print("\nFinal metrics:")
        summary = router.metrics.summary()
        print(f"  Cache hit rate: {summary.get('cache_hit_rate', 'N/A')}")
        latency = summary.get("latency", {}).get("fast_talker", {})
        if "total_response" in latency:
            print(f"  Avg response: {latency['total_response']['avg_ms']:.1f}ms")
        print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nExited.")
        sys.exit(0)

