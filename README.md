# HH Goa 2026 — Voice-Enabled RAG Pipeline

A voice-enabled Retrieval-Augmented Generation (RAG) system that transcribes spoken questions, retrieves relevant context from MSMARCO-XI, and generates answers end-to-end.

## Pipeline

```
Voice Input → ElevenLabs Scribe v2 (WebSocket STT)
    → Embedding (ONNX MiniLM-L6-v2, <5ms)
    → Binary FAISS Search (Hamming, <1.5ms)
    → Float16 Rescore (top-100 → top-5, <1ms)
    → SQLite Payload Fetch (<0.5ms)
    → SmolLM2-135M via llama.cpp (TTFT ~15ms)
    → Answer (streamed via SSE)
```

**Total retrieval latency: < 6ms** | **Total pipeline: < 200ms**

## Architecture

- **Dataset**: [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) (500K English passages)
- **Embedding**: all-MiniLM-L6-v2 (ONNX INT8, 384-dim)
- **Vector DB**: FAISS IndexBinaryFlat (binary quantization, 48 bytes/vector)
- **LLM**: SmolLM2-135M-Instruct via llama.cpp (Q4_K_M, ~85MB)
- **STT**: ElevenLabs Scribe v2 Realtime (WebSocket streaming)
- **Chunking**: 4 strategies (PassageAsChunk, SlidingWindow, SemanticSentence, Hierarchical)
- **Guardrails**: Input safety, off-topic detection, output grounding check
- **Harness**: Circuit breaker, timeout enforcement, structured I/O

## Quick Start

### 1. Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### 2. Ingest Dataset

```bash
# Download and sample 500K passages from MSMARCO-XI
python -m ingestion.download_dataset --n-passages 500000

# Build FAISS binary index + float16 vectors
python -m ingestion.build_index --strategy passage_as_chunk

# Build SQLite payload database
python -m ingestion.build_payload_db
```

### 3. Run Server

```bash
# Start llama.cpp server (in background)
llama-server \
  --model models/smollm2-135m-instruct-q4_k_m.gguf \
  --host 0.0.0.0 --port 8081 \
  --threads 4 --ctx-size 1024 &

# Start FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Docker

```bash
docker compose build
docker compose up -d
```

### 5. Test

```bash
# Health check
curl http://localhost:8000/health

# Text query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "What is retrieval augmented generation?"}'
```

## Benchmark

```bash
# Retrieval-only (target: P100 < 6ms)
python benchmark.py --mode retrieval --queries 100

# Full pipeline (target: P100 < 200ms)
python benchmark.py --mode pipeline --queries 50

# Both
python benchmark.py --mode all --queries 100
```

## Chunking Strategies

| Strategy | Description | Use Case |
|---|---|---|
| PassageAsChunk | Use MSMARCO passages as-is | Production (optimal for pre-chunked data) |
| SlidingWindow | Fixed-size with overlap | Long documents |
| SemanticSentence | Sentence-boundary splitting | Preserving semantic coherence |
| Hierarchical | Two-level (passage + group) | Broad + precise retrieval |

## Guardrails

1. **Input Safety**: Keyword blocklist, length validation
2. **Relevance**: Off-topic detection via corpus centroid similarity
3. **Grounding**: Token-overlap check between LLM output and retrieved context

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | Text query → JSON response with answer, passages, metrics |
| `/query/stream` | POST | Text query → SSE streaming response |
| `/voice-stream` | WebSocket | Streaming audio → STT → RAG → tokens |
| `/health` | GET | Component readiness check |

## Latency Budget

| Stage | Budget | Strategy |
|---|---|---|
| Embedding | < 5ms | ONNX INT8 all-MiniLM-L6-v2 |
| Binary Search | < 1.5ms | FAISS IndexBinaryFlat, POPCNT |
| Rescore | < 1ms | Float16 memmap, dot product |
| Payload Fetch | < 0.5ms | SQLite WAL mode |
| **Total Retrieval** | **< 6ms** | |
| LLM TTFT | ~15ms | SmolLM2-135M Q4_K_M |
| **Total Pipeline** | **< 200ms** | |
