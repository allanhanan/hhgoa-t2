"""Pydantic models for structured I/O across the pipeline."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ── Retrieval ─────────────────────────────────────────────────────────

class PassageResult(BaseModel):
    """A single retrieved passage with its score."""
    id: int
    text: str
    score: float
    query_type: str = ""
    source_query: str = ""


class RetrievalResult(BaseModel):
    """Output from the retrieval stage."""
    passages: list[PassageResult]
    embed_ms: float = 0.0
    search_ms: float = 0.0
    rescore_ms: float = 0.0
    payload_ms: float = 0.0
    total_ms: float = 0.0


# ── Pipeline ──────────────────────────────────────────────────────────

class PipelineMetrics(BaseModel):
    """Per-stage latency metrics collected through the pipeline."""
    guardrail_input_ms: float = 0.0
    embed_ms: float = 0.0
    search_ms: float = 0.0
    rescore_ms: float = 0.0
    payload_ms: float = 0.0
    guardrail_context_ms: float = 0.0
    answer_ms: float = 0.0
    generate_ttft_ms: float = 0.0
    generate_total_ms: float = 0.0
    guardrail_output_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def retrieval_ms(self) -> float:
        return self.embed_ms + self.search_ms + self.rescore_ms + self.payload_ms


class PipelineResult(BaseModel):
    """Final output of the full RAG pipeline."""
    answer: str = ""
    passages: list[PassageResult] = Field(default_factory=list)
    metrics: PipelineMetrics = Field(default_factory=PipelineMetrics)
    grounded: bool = True
    error: str | None = None


# ── API request / response ────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Incoming text query."""
    text: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """API response for /query endpoint."""
    answer: str
    passages: list[PassageResult]
    metrics: PipelineMetrics
    grounded: bool = True
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    index_loaded: bool = False
    embedding_model_loaded: bool = False
    qa_model_loaded: bool = False
    llm_available: bool = False
