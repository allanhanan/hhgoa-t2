"""Configuration management for VoiceAgentRAG."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class VORConfig(BaseSettings):
    """Central configuration for the VoiceAgentRAG system."""

    model_config = {"env_prefix": "VOR_", "env_file": ".env", "extra": "ignore"}

    # LLM settings
    llm_provider: Literal["openai", "anthropic", "ollama", "gemini", "groq"] = "groq"
    llm_model: str = "llama-3.1-8b-instant"  # Fastest Groq model: ~150ms vs ~6000ms for 70B
    llm_api_key: str = ""
    llm_base_url: str | None = None
    llm_temperature: float = 0.3
    llm_max_tokens: int = 64  # Cap response length to reduce latency

    # Embedding settings
    embedding_provider: Literal["openai", "ollama", "sentence-transformers", "onnx"] = "onnx"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Vector store settings
    vector_store_provider: Literal["faiss", "qdrant"] = "qdrant"
    faiss_index_path: Path = Path("data/faiss_index")
    retrieval_latency_ms: float = 0  # Simulated retrieval latency for benchmarking

    # Qdrant settings (when vector_store_provider="qdrant")
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "voice_rag"

    # Semantic cache settings
    cache_max_size: int = 2000
    cache_ttl_seconds: float = 900.0  # 15 min: longer TTL = more cache hits = faster responses
    cache_similarity_threshold: float = 0.35  # Lower = more cache hits on rephrased queries

    # Slow thinker settings
    prediction_strategy: Literal["keyword", "llm"] = "llm"
    max_predictions: int = 5
    prefetch_top_k: int = 10
    slow_thinker_rate_limit: float = 0.5  # min seconds between predictions

    # Fast talker settings
    fast_talker_max_context_chunks: int = 2  # 2 chunks << 10: cuts prompt tokens, major speed-up
    fast_talker_fallback_to_retrieval: bool = True

    # Guardrails
    guardrails_enabled: bool = True
    guardrail_min_context_chunks: int = 1
    guardrail_min_relevance_score: float = 0.25
    guardrail_refusal_message: str = "I can't answer that from the available context."
    guardrail_off_topic_terms: list[str] = Field(default_factory=list)

    # Conversation stream settings
    conversation_window_size: int = 10

    # Document chunking
    chunking_strategy: Literal["sentence", "semantic", "fixed-token", "metadata", "parent-child"] = "sentence"
    chunk_size: int = 512
    chunk_overlap: int = 50
    parent_chunk_size: int = 2048

    # Voice settings
    stt_provider: Literal["elevenlabs", "whisper", "deepgram", "openai"] = "elevenlabs"
    tts_provider: Literal["none", "openai", "elevenlabs", "edge"] = "none"
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model: str = "eleven_multilingual_v2"
    elevenlabs_stt_model: str = "scribe_v2"
    whisper_model: str = "base.en"
    sample_rate: int = 16000
    vad_aggressiveness: int = 2

    # Gemini / Vertex AI settings
    gemini_api_key: str = ""
    vertex_project: str = ""
    vertex_location: str = "us-central1"
