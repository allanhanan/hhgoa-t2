"""Central configuration for the RAG pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

INDEX_PATH = str(DATA_DIR / "index.faiss_binary")
FLOAT16_PATH = str(DATA_DIR / "float16_vectors.npy")
PAYLOAD_DB = str(DATA_DIR / "payloads.db")
QUERIES_PATH = str(DATA_DIR / "benchmark_queries.json")

# ── Latency budgets (ms) ──────────────────────────────────────────────
LATENCY_BUDGET_MS = 200     # Full pipeline timeout budget (ms)
RETRIEVAL_BUDGET_MS = 6       # Embed + search + rescore + payload

# ── Retrieval tuning ──────────────────────────────────────────────────
TOP_K_BINARY = 8              # Binary search returns top-8 candidates for instant <0.1ms rescore
TOP_K_FINAL = 5               # After rescore, return top-5 passages
EMBEDDING_DIM = 384           # all-MiniLM-L6-v2 dimension
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_SIZE = 500_000          # Number of passages to index

# ── Extractive QA ─────────────────────────────────────────────────────
QA_ONNX_MODEL = str(MODELS_DIR / "onnx" / "minilm_qa.onnx")
QA_TOKENIZER_NAME = "deepset/minilm-uncased-squad2"
QA_CONFIDENCE_THRESHOLD = 0.01     # Minimum confidence to accept extracted span
HEURISTIC_CONFIDENCE = 0.7         # Tier 1 heuristic confidence threshold
MAX_TOKEN_LIMIT = 256              # Max token limit for encoder/QA sequence limits
QA_MAX_SPAN_TOKENS = 256           # Max extracted answer span length in tokens

# ── STT ───────────────────────────────────────────────────────────────
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# ── Guardrails ────────────────────────────────────────────────────────
SAFETY_MAX_QUERY_LEN = 500
RELEVANCE_THRESHOLD = 0.35    # Max top passage similarity threshold (0.35)
GROUNDING_OVERLAP_THRESHOLD = 0.3

# ── Dataset ───────────────────────────────────────────────────────────
DATASET_NAME = "ai4bharat/MSMARCO-XI"
DATASET_LANG = "default"      # Uses default config; filter English rows by source_lang
