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
LATENCY_BUDGET_MS = 3000      # Full pipeline timeout budget (ms)
RETRIEVAL_BUDGET_MS = 6       # Embed + search + rescore + payload

# ── Retrieval tuning ──────────────────────────────────────────────────
TOP_K_BINARY = 8              # Binary search returns top-8 candidates for instant <0.1ms rescore
TOP_K_FINAL = 5               # After rescore, return top-5 passages
EMBEDDING_DIM = 384           # all-MiniLM-L6-v2 dimension
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_SIZE = 500_000          # Number of passages to index

# ── LLM ───────────────────────────────────────────────────────────────
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:2000/v1")
LLAMA_CPP_URL = os.getenv("LLAMA_CPP_URL", "http://localhost:8081")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "SupraLabs/Supra2-100M-Instruct")
LLM_MAX_TOKENS = 200
LLM_TEMPERATURE = 0.1
LLM_REPEAT_PENALTY = 1.3

SYSTEM_PROMPT = (
    "Answer in one sentence using ONLY the provided context. "
    "If the answer is not in the context, say: "
    "'I cannot answer from the provided information.'"
)

# ── STT ───────────────────────────────────────────────────────────────
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# ── Guardrails ────────────────────────────────────────────────────────
SAFETY_MAX_QUERY_LEN = 500
RELEVANCE_THRESHOLD = 0.0    # Cosine sim vs corpus centroid (permissive for all queries)
GROUNDING_OVERLAP_THRESHOLD = 0.3

# ── Groq fallback ─────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Dataset ───────────────────────────────────────────────────────────
DATASET_NAME = "ai4bharat/MSMARCO-XI"
DATASET_LANG = "default"      # Uses default config; filter English rows by source_lang
