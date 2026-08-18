"""FastAPI application — main server with /query, /voice-stream, /health endpoints."""
from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from app.config import DATA_DIR
from app.models import QueryRequest, QueryResponse, HealthResponse
from app.retriever import vector_db, payload_store
from app.retriever.rescorer import load_vectors as load_rescore_vectors, is_loaded as rescore_loaded
from app.embedder.encoder import warmup as warmup_encoder
from app.guardrails.relevance import set_centroid, compute_centroid_from_vectors
from app.harness.orchestrator import run_pipeline
from app.harness.timeout import with_timeout

import numpy as np

logger = logging.getLogger("rag-pipeline")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load all models and indices into memory."""
    logger.info("Loading FAISS binary index...")
    vector_db.load_index()
    logger.info(f"  Index loaded: {vector_db.index_size()} vectors")

    logger.info("Loading float16 rescore vectors...")
    vecs = load_rescore_vectors()
    logger.info(f"  Rescore vectors: {vecs.shape}")

    logger.info("Loading SQLite payload store...")
    payload_store.connect()

    logger.info("Warming up embedding model...")
    warmup_encoder()

    # Compute corpus centroid for relevance guardrail
    logger.info("Computing corpus centroid for relevance guardrail...")
    centroid_path = str(DATA_DIR / "corpus_centroid.npy")
    try:
        centroid = np.load(centroid_path)
        set_centroid(centroid)
        logger.info("  Centroid loaded from disk")
    except FileNotFoundError:
        logger.warning("  No centroid file found; relevance guardrail will be permissive")

    logger.info("RAG pipeline ready!")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="HH Goa 2026 — Voice-Enabled RAG Pipeline",
    description="Voice → STT → Retrieval (FAISS binary) → LLM → Answer",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Text query → RAG pipeline → JSON response with metrics."""
    result = await with_timeout(
        run_pipeline(request.text, top_k=request.top_k)
    )
    return QueryResponse(
        answer=result.answer,
        passages=result.passages,
        metrics=result.metrics,
        grounded=result.grounded,
        error=result.error,
    )


@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """Text query → RAG pipeline → SSE streaming response."""

    async def event_stream():
        result = await run_pipeline(request.text, top_k=request.top_k)

        # Send passages first
        yield f"data: {json.dumps({'type': 'passages', 'data': [p.model_dump() for p in result.passages]})}\n\n"

        # Send answer
        yield f"data: {json.dumps({'type': 'answer', 'data': result.answer})}\n\n"

        # Send metrics
        yield f"data: {json.dumps({'type': 'metrics', 'data': result.metrics.model_dump()})}\n\n"

        # Send grounding status
        yield f"data: {json.dumps({'type': 'grounding', 'data': result.grounded})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.websocket("/voice-stream")
async def voice_stream_endpoint(ws: WebSocket):
    """WebSocket endpoint for streaming voice → RAG pipeline.

    Client sends binary PCM audio chunks.
    Server relays to ElevenLabs STT, runs pipeline on final transcript,
    and streams back tokens.
    """
    await ws.accept()
    try:
        from app.stt.elevenlabs import stream_transcribe

        # Collect audio chunks from client
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        async def audio_iterator():
            while True:
                chunk = await audio_queue.get()
                if chunk is None:  # Sentinel
                    break
                yield chunk

        # Start STT in background
        stt_task = None
        final_text = ""

        while True:
            try:
                data = await asyncio.wait_for(ws.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                break

            if "bytes" in data:
                await audio_queue.put(data["bytes"])
            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "end_audio":
                    await audio_queue.put(None)  # Signal end
                    break
                elif msg.get("type") == "text_query":
                    # Direct text query via WebSocket
                    final_text = msg.get("text", "")
                    break

        # If we have audio, transcribe it
        if not final_text:
            async for transcript in stream_transcribe(audio_iterator()):
                await ws.send_json({"type": "transcript", **transcript})
                if transcript.get("is_final"):
                    final_text = transcript.get("text", "")
                    break

        # Run pipeline on final text
        if final_text:
            result = await run_pipeline(final_text)
            await ws.send_json({
                "type": "result",
                "answer": result.answer,
                "passages": [p.model_dump() for p in result.passages],
                "metrics": result.metrics.model_dump(),
                "grounded": result.grounded,
                "error": result.error,
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """Health check — reports component readiness."""
    from app.generator.local_llm import health_check as llm_health

    return HealthResponse(
        status="ok",
        index_loaded=vector_db.is_loaded(),
        embedding_model_loaded=True,  # Loaded during startup
        llm_available=await llm_health(),
    )
