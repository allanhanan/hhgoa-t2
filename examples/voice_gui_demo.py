#!/usr/bin/env python3
"""Web UI Voice Agent Demo: microphone to ElevenLabs STT to RAG answers with modern Web UI.

Flow:
    Microphone input -> ElevenLabs STT -> MemoryRouter (Qdrant + Groq)
    Serves web UI on http://localhost:8000 with live time, listening animation & results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_optimized_rag import MemoryRouter, VORConfig
from voice_optimized_rag.voice.audio_stream import AudioStream
from voice_optimized_rag.voice.stt import create_stt
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("voice_gui")
STATIC_DIR = Path(__file__).parent / "static"

connected_websockets: set[web.WebSocketResponse] = set()


async def broadcast(message: dict[str, object]) -> None:
    """Broadcast a JSON message to all connected WebSocket clients."""
    if not connected_websockets:
        return
    payload = json.dumps(message)
    dead_sockets = set()
    for ws in list(connected_websockets):
        try:
            await ws.send_str(payload)
        except Exception:
            dead_sockets.add(ws)
    connected_websockets.difference_update(dead_sockets)


async def process_utterance_and_respond(utterance: bytes, stt, router: MemoryRouter, config: VORConfig) -> None:
    """Process an audio utterance from browser mic or server mic, run RAG, and broadcast response."""
    duration_s = len(utterance) / (config.sample_rate * 2)
    await broadcast({
        "type": "stt_sending",
        "bytes": len(utterance),
        "duration": duration_s,
    })
    await broadcast({"type": "status", "state": "processing", "text": "Transcribing speech..."})

    stt_start = time.perf_counter()
    try:
        text = await stt.transcribe(utterance, config.sample_rate)
    except Exception as err:
        stt_ms = (time.perf_counter() - stt_start) * 1000
        logger.error(f"STT Error: {err}")
        print(f"[VoiceGUI] STT Error ({stt_ms:.0f}ms): {err}")
        await broadcast({"type": "status", "state": "error", "text": f"STT Error: {err}"})
        await broadcast({
            "type": "answer",
            "text": f"⚠️ Speech-to-Text Error ({err}).\n\nNote: ElevenLabs returned an API error for the current key in .env. You can update VOR_ELEVENLABS_API_KEY in .env or type a question directly in the text box below!",
            "timings": {"stt_ms": stt_ms, "rag_ms": 0.0},
            "hit_rate": 0.0,
        })
        return
    stt_ms = (time.perf_counter() - stt_start) * 1000

    if not text.strip():
        await broadcast({"type": "status", "state": "listening", "text": "No speech detected. Speak again."})
        return

    await broadcast({"type": "transcript", "text": text.strip()})
    await broadcast({"type": "status", "state": "processing", "text": "Searching knowledge base..."})

    query_start = time.perf_counter()
    response_parts: list[str] = []
    async for chunk in router.query_stream(text.strip()):
        response_parts.append(chunk)
        await broadcast({"type": "answer_chunk", "chunk": chunk})

    response = "".join(response_parts)
    query_ms = (time.perf_counter() - query_start) * 1000
    hit_rate = router.metrics.cache_hit_rate

    await broadcast({
        "type": "answer",
        "text": response,
        "timings": {"stt_ms": stt_ms, "rag_ms": query_ms},
        "hit_rate": hit_rate,
    })


async def process_text_query(text: str, router: MemoryRouter) -> None:
    """Process a text query sent directly from Web UI."""
    await broadcast({"type": "status", "state": "processing", "text": "Searching knowledge base..."})

    query_start = time.perf_counter()
    response_parts: list[str] = []
    async for chunk in router.query_stream(text.strip()):
        response_parts.append(chunk)
        await broadcast({"type": "answer_chunk", "chunk": chunk})

    response = "".join(response_parts)
    query_ms = (time.perf_counter() - query_start) * 1000
    hit_rate = router.metrics.cache_hit_rate

    await broadcast({
        "type": "answer",
        "text": response,
        "timings": {"stt_ms": 0.0, "rag_ms": query_ms},
        "hit_rate": hit_rate,
    })


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """Handle WebSocket connection from Web UI."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    connected_websockets.add(ws)
    await ws.send_str(json.dumps({"type": "status", "state": "listening", "text": "Click Mic to Speak"}))

    app_stt = request.app["stt"]
    app_router = request.app["router"]
    app_config = request.app["config"]

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                audio_bytes = msg.data
                await process_utterance_and_respond(audio_bytes, app_stt, app_router, app_config)
            elif msg.type == web.WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                    if payload.get("type") == "text_query":
                        text = payload.get("text", "")
                        if text:
                            await process_text_query(text, app_router)
                except Exception as err:
                    logger.warning(f"WebSocket text parse error: {err}")
    finally:
        connected_websockets.remove(ws)
    return ws


async def index_handler(request: web.Request) -> web.FileResponse:
    """Serve the Web UI index.html."""
    return web.FileResponse(STATIC_DIR / "index.html")


async def _process_partial(utterance: bytes, stt, router: MemoryRouter, config: VORConfig) -> None:
    """Background task to transcribe a partial utterance and trigger speculative retrieval."""
    try:
        text = await stt.transcribe(utterance, config.sample_rate)
        if text.strip():
            await router.stream.emit_partial(text.strip())
    except Exception as e:
        logger.debug(f"Partial STT failed: {e}")


async def voice_agent_loop(audio: AudioStream, stt, router: MemoryRouter, config: VORConfig) -> None:
    """Background loop reading microphone frames and running RAG."""
    await broadcast({"type": "status", "state": "listening", "text": "Listening for speech..."})

    try:
        async for is_final, utterance in audio.listen(yield_partials=True):
            if not is_final:
                # Fire and forget partial transcription for mid-sentence speculative retrieval
                asyncio.create_task(_process_partial(utterance, stt, router, config))
                continue

            duration_s = len(utterance) / (config.sample_rate * 2)
            await broadcast({
                "type": "stt_sending",
                "bytes": len(utterance),
                "duration": duration_s,
            })
            await broadcast({"type": "status", "state": "processing", "text": "Transcribing speech..."})

            stt_start = time.perf_counter()
            text = await stt.transcribe(utterance, config.sample_rate)
            stt_ms = (time.perf_counter() - stt_start) * 1000

            if not text.strip():
                await broadcast({"type": "status", "state": "listening", "text": "Listening for speech..."})
                continue

            await broadcast({"type": "transcript", "text": text.strip()})
            await broadcast({"type": "status", "state": "processing", "text": "Searching knowledge base..."})

            query_start = time.perf_counter()
            response_parts: list[str] = []
            async for chunk in router.query_stream(text):
                response_parts.append(chunk)
                await broadcast({"type": "answer_chunk", "chunk": chunk})

            response = "".join(response_parts)
            query_ms = (time.perf_counter() - query_start) * 1000
            hit_rate = router.metrics.cache_hit_rate

            await broadcast({
                "type": "answer",
                "text": response,
                "timings": {"stt_ms": stt_ms, "rag_ms": query_ms},
                "hit_rate": hit_rate,
            })
            await broadcast({"type": "status", "state": "listening", "text": "Listening for speech..."})

    except (asyncio.CancelledError, KeyboardInterrupt):
        pass


async def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Agent Web UI Demo")
    parser.add_argument("--port", type=int, default=8000, help="Port for Web UI server")
    parser.add_argument("--provider", default="groq", choices=["openai", "anthropic", "ollama", "gemini", "groq"])
    parser.add_argument("--stt", default="elevenlabs", choices=["elevenlabs", "openai", "whisper"])
    args = parser.parse_args()

    config = VORConfig(
        llm_provider=args.provider,
        stt_provider=args.stt,
        vector_store_provider="qdrant",
        qdrant_collection="msmarco_xi_test",
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        # Fastest Groq model: ~150ms vs ~6000ms for 70B; switch to llama-3.3-70b-versatile for quality
        llm_model="llama-3.1-8b-instant" if args.provider == "groq" else None,
        llm_base_url="https://api.groq.com/openai/v1" if args.provider == "groq" else None,
        llm_max_tokens=256,             # Cap response length: shorter = faster
        fast_talker_max_context_chunks=3,  # 3 chunks cuts prompt size ~70%
        cache_ttl_seconds=900.0,           # 15 min cache TTL = more cache hits
        cache_similarity_threshold=0.35,   # More permissive cache matching
        guardrails_enabled=True,           # Strictly enforce grounded RAG guardrails
    )

    loop = asyncio.get_running_loop()

    def silence_windows_disconnect_noise(loop_inst, context):
        exception = context.get("exception")
        if isinstance(exception, ConnectionResetError):
            return
        loop_inst.default_exception_handler(context)

    loop.set_exception_handler(silence_windows_disconnect_noise)

    def on_energy(energy: float, threshold: float, is_speech: bool):
        asyncio.run_coroutine_threadsafe(
            broadcast({
                "type": "energy",
                "energy": energy,
                "threshold": threshold,
                "is_speech": is_speech,
            }),
            loop,
        )

    router = MemoryRouter(config)
    audio = AudioStream(
        sample_rate=config.sample_rate,
        vad_aggressiveness=config.vad_aggressiveness,
        energy_callback=on_energy,
    )
    stt = create_stt(args.stt, api_key=config.elevenlabs_api_key, model=config.elevenlabs_stt_model)

    await router.start(log_level="INFO")

    app = web.Application()
    app["stt"] = stt
    app["router"] = router
    app["config"] = config
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_static("/static/", STATIC_DIR)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", args.port)
    await site.start()

    print("=" * 60)
    print("  Voice-Optimized RAG - Web UI Assistant")
    print(f"  Open UI in your browser: http://127.0.0.1:{args.port}")
    print("=" * 60)

    # Start audio processing in background
    agent_task = asyncio.create_task(voice_agent_loop(audio, stt, router, config))

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nStopping Web UI server...")
    finally:
        agent_task.cancel()
        await runner.cleanup()
        await router.stop()
        print("Server stopped cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        sys.exit(0)
