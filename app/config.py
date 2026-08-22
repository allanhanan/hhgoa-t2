"""Central configuration for the RAG pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Cap threading overhead across OpenMP / BLAS / PyTorch / ONNX to prevent core contention
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))

INDEX_PATH = os.getenv("INDEX_PATH", str(DATA_DIR / "index.faiss_binary"))
IVF_INDEX_PATH = os.getenv("IVF_INDEX_PATH", str(DATA_DIR / "index.faiss_ivf.bin"))
FLOAT16_PATH = os.getenv("FLOAT16_PATH", str(DATA_DIR / "float16_vectors.npy"))
PAYLOAD_DB = os.getenv("PAYLOAD_DB", str(DATA_DIR / "payloads.db"))
QUERIES_PATH = os.getenv("QUERIES_PATH", str(DATA_DIR / "benchmark_queries.json"))
BM25_INDEX_DIR = os.getenv("BM25_INDEX_DIR", str(DATA_DIR / "bm25_index"))
CENTROID_PATH = os.getenv("CENTROID_PATH", str(DATA_DIR / "corpus_centroid.npy"))

# ── Latency budgets (ms) ──────────────────────────────────────────────
LATENCY_BUDGET_MS = 100     # Full pipeline timeout budget (ms)
RETRIEVAL_BUDGET_MS = 20       # Embed + search + rescore + payload

# ── Retrieval tuning ──────────────────────────────────────────────────
FAISS_THREADS = 1              # Number of threads for FAISS search
USE_IVF = True                 # Set to True to use FAISS IndexIVFScalarQuantizer (SQ8) instead of IndexBinaryFlat
IVF_NLIST = 8192               # Number of Voronoi cells for IVF
IVF_NPROBE = 1                 # Number of cells queried during IVF search
ANN_TOP_K = 8                 # Number of candidates returned by ANN search for FP16 rescoring
TOP_K_BINARY = 32              # Binary search candidates (legacy fallback)
TOP_K_FINAL = 5               # After rescore, return top-5 passages
EMBEDDING_DIM = 384           # all-MiniLM-L6-v2 dimension
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_SIZE = None          # Number of passages to index

# ── Device & Hardware Acceleration ──────────────────────────────────
DEVICE = os.getenv("DEVICE", "auto").lower()  # "auto", "cuda", "dml", "cpu"


def get_onnx_providers(model_path: str | Path | None = None) -> list[str]:
    """Get prioritized list of ONNX Runtime execution providers.

    INT8 models (minilm_int8.onnx, minilm_qa_int8.onnx) are dynamically quantized
    for CPU (AVX2/VNNI). Running CPU-quantized INT8 models on CUDA causes ONNX Runtime
    to insert 80-170+ host-to-device Memcpy nodes per pass, degrading latency.
    Therefore, INT8 models use CPUExecutionProvider (<3ms latency), while FP32/FP16 models
    utilize CUDA / DirectML GPU acceleration when available.
    """
    if DEVICE == "cpu":
        return ["CPUExecutionProvider"]

    if model_path is not None:
        model_str = str(model_path).lower()
        if "int8" in model_str or "quant" in model_str:
            return ["CPUExecutionProvider"]

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = []
        if DEVICE in ("cuda", "auto") and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if DEVICE in ("dml", "auto") and "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers
    except Exception:
        return ["CPUExecutionProvider"]


def get_onnx_session_options(model_path: str | Path | None = None):
    """Create ONNX SessionOptions tuned for the active execution provider.

    When GPU handles inference, CPU threads are minimized (they only do I/O).
    When CPU-only, uses 2 intra-op threads for parallelism.
    """
    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    providers = get_onnx_providers(model_path)
    if any(p != "CPUExecutionProvider" for p in providers):
        # GPU active: minimize CPU thread overhead
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
    else:
        # CPU-only: use 2 threads for parallelism
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return sess_opts

# ── Extractive QA ─────────────────────────────────────────────────────
QA_ONNX_MODEL = os.getenv("QA_ONNX_MODEL", str(MODELS_DIR / "onnx" / "minilm_qa_int8.onnx"))
QA_TOKENIZER_NAME = "deepset/minilm-uncased-squad2"
QA_CONFIDENCE_THRESHOLD = 0.15     # Minimum confidence to accept extracted span
QA_NULL_THRESHOLD: float = 0.0     # TUNE: sweep against a labeled dev set
QA_MARGIN_THRESHOLD: float = 0.05  # TUNE: sweep against a labeled dev set
HEURISTIC_CONFIDENCE = 0.3         # Tier 1 heuristic confidence threshold
MAX_TOKEN_LIMIT = 256              # Max token limit for encoder/QA sequence limits
QA_MAX_SPAN_TOKENS = 256           # Max extracted answer span length in tokens
GENERATION_BACKEND = "local"       # Clamps eval pipeline workers to 1 to prevent CPU thread contention
GENERATION_MODEL = "minilm-qa"
LOCAL_GENERATION_MODEL = "minilm-qa"

# ── STT ───────────────────────────────────────────────────────────────
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# ── Guardrails ────────────────────────────────────────────────────────
SAFETY_MAX_QUERY_LEN = 256
RELEVANCE_THRESHOLD = 0.33    # Max top passage similarity threshold (0.35)
RELEVANCE_MARGIN_THRESHOLD: float = 0.00   # TUNE: sweep against labeled queries
RELEVANCE_MIN_ABS_SCORE: float = 0.42     # TUNE: reject near-zero floor regardless of margin
GROUNDING_OVERLAP_THRESHOLD = 0.3

# ── Dataset ───────────────────────────────────────────────────────────
DATASET_NAME = "ai4bharat/MSMARCO-XI"
DATASET_LANG = "default"      # Uses default config; filter English rows by source_lang

# ── Query Cache ───────────────────────────────────────────────────────
QUERY_CACHE_ENABLED = True
QUERY_CACHE_TTL_SEC = 3600
QUERY_CACHE_MAX_SIZE = 10000
