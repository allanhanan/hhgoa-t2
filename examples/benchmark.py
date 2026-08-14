#!/usr/bin/env python3
"""Benchmark: Compare traditional RAG vs Voice-Optimized RAG (dual-agent).

Measures latency, cache hit rates, and response quality across multi-turn
conversations. Generates a comparison report.

Usage:
    VOR_LLM_API_KEY=sk-... python examples/benchmark.py --docs path/to/docs/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_optimized_rag import MemoryRouter, VORConfig
from voice_optimized_rag.core.conversation_stream import EventType, StreamEvent
from voice_optimized_rag.retrieval.embeddings import create_embedding_provider
from voice_optimized_rag.retrieval.vector_store import FAISSVectorStore
from voice_optimized_rag.llm.base import create_llm
from voice_optimized_rag.utils.metrics import MetricsCollector, Timer

# Sample multi-turn conversations for benchmarking
SAMPLE_CONVERSATIONS = [
    {
        "topic": "Product Features",
        "turns": [
            "What are the main features of the product?",
            "Tell me more about the API integration capabilities.",
            "What programming languages are supported?",
            "How does the authentication work?",
            "What are the rate limits?",
        ],
    },
    {
        "topic": "Pricing and Plans",
        "turns": [
            "What pricing plans are available?",
            "What's included in the enterprise plan?",
            "Are there any discounts for annual billing?",
            "What happens if I exceed my usage limits?",
            "Can I upgrade my plan at any time?",
        ],
    },
    {
        "topic": "Technical Support",
        "turns": [
            "How do I get started with the API?",
            "I'm getting a 401 error, what should I check?",
            "How do I handle pagination in the response?",
            "What's the recommended retry strategy?",
            "Are there any webhooks available?",
        ],
    },
]

FIXED_TEST_QUERIES = [
    turn
    for conversation in SAMPLE_CONVERSATIONS
    for turn in conversation["turns"]
]


def latency_percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50_ms": 0.0, "p70_ms": 0.0, "p100_ms": 0.0}
    sorted_values = sorted(values)
    def pick(percentile: float) -> float:
        idx = int((len(sorted_values) - 1) * (percentile / 100))
        return round(sorted_values[idx], 1)
    return {
        "p50_ms": pick(50),
        "p70_ms": pick(70),
        "p100_ms": round(sorted_values[-1], 1),
    }


async def benchmark_traditional_rag(
    config: VORConfig,
    conversations: list[dict],
) -> dict:
    """Benchmark traditional RAG (retrieve on every query)."""
    print("\n--- Traditional RAG (baseline) ---")
    metrics = MetricsCollector()
    llm = create_llm(config)
    embeddings = create_embedding_provider(config)
    vector_store = FAISSVectorStore(
        dimension=embeddings.dimension,
        index_path=config.faiss_index_path if config.faiss_index_path.exists() else None,
    )

    latencies: list[float] = []

    for convo in conversations:
        print(f"\n  Topic: {convo['topic']}")
        for turn in convo["turns"]:
            start = time.perf_counter()

            # Traditional: embed → search → generate (sequential)
            query_emb = await embeddings.embed_single(turn)
            results = vector_store.search(query_emb, top_k=5)
            context = "\n\n".join(r.text for r in results) if results else ""
            response = await llm.generate(turn, context=context)

            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            print(f"    [{elapsed_ms:6.0f}ms] {turn[:50]}...")

    return {
        "method": "Traditional RAG",
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        **latency_percentiles(latencies),
        "total_queries": len(latencies),
        "all_latencies": [round(l, 1) for l in latencies],
    }


async def benchmark_dual_agent(
    config: VORConfig,
    conversations: list[dict],
) -> dict:
    """Benchmark Voice-Optimized RAG (dual-agent with prefetching)."""
    print("\n--- Voice-Optimized RAG (dual-agent) ---")
    router = MemoryRouter(config)
    await router.start(log_level="WARNING")

    latencies: list[float] = []
    cache_hits_per_turn: list[bool] = []

    for convo in conversations:
        print(f"\n  Topic: {convo['topic']}")
        # Clear cache between topics to simulate topic shifts
        await router.cache.clear()

        for i, turn in enumerate(convo["turns"]):
            # Small delay to let slow thinker work (simulates speech time)
            if i > 0:
                await asyncio.sleep(0.5)

            hits_before = router.metrics.get_counter("cache_hit")
            start = time.perf_counter()

            response_parts = []
            async for chunk in router.query_stream(turn):
                response_parts.append(chunk)

            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            hits_after = router.metrics.get_counter("cache_hit")
            was_hit = hits_after > hits_before
            cache_hits_per_turn.append(was_hit)

            hit_str = "HIT " if was_hit else "MISS"
            print(f"    [{elapsed_ms:6.0f}ms] [{hit_str}] {turn[:50]}...")

    await router.stop()

    hit_rate = sum(cache_hits_per_turn) / len(cache_hits_per_turn) if cache_hits_per_turn else 0

    return {
        "method": "Voice-Optimized RAG",
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        **latency_percentiles(latencies),
        "total_queries": len(latencies),
        "cache_hits": sum(cache_hits_per_turn),
        "cache_misses": len(cache_hits_per_turn) - sum(cache_hits_per_turn),
        "cache_hit_rate": f"{hit_rate:.1%}",
        "all_latencies": [round(l, 1) for l in latencies],
        "full_metrics": router.metrics.summary(),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark: Traditional RAG vs Dual-Agent")
    parser.add_argument("--docs", type=Path, help="Document directory (uses existing index if available)")
    parser.add_argument("--provider", default="groq", choices=["openai", "anthropic", "ollama", "gemini", "groq"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", type=Path, default=None, help="Save results to JSON")
    args = parser.parse_args()

    config_kwargs = {"llm_provider": args.provider}
    if args.model:
        config_kwargs["llm_model"] = args.model
    if args.api_key:
        config_kwargs["llm_api_key"] = args.api_key
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
        config_kwargs["embedding_provider"] = "sentence-transformers"
        config_kwargs["embedding_model"] = "all-MiniLM-L6-v2"
        config_kwargs["embedding_dimension"] = 384

    config = VORConfig(**config_kwargs)

    # Ingest docs if needed
    if args.docs and args.docs.is_dir() and not config.faiss_index_path.exists():
        print(f"Ingesting documents from {args.docs}...")
        router = MemoryRouter(config)
        count = await router.ingest_directory(args.docs)
        router.save_index()
        print(f"Ingested {count} chunks\n")

    print("=" * 60)
    print("  Voice-Optimized RAG — Benchmark")
    print(f"  Conversations: {len(SAMPLE_CONVERSATIONS)}")
    print(f"  Total turns: {sum(len(c['turns']) for c in SAMPLE_CONVERSATIONS)}")
    print("=" * 60)

    # Run benchmarks
    traditional = await benchmark_traditional_rag(config, SAMPLE_CONVERSATIONS)
    dual_agent = await benchmark_dual_agent(config, SAMPLE_CONVERSATIONS)

    # Report
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n  {'Metric':<30} {'Traditional':>15} {'Dual-Agent':>15}")
    print(f"  {'-'*60}")
    print(f"  {'Avg Latency (ms)':<30} {traditional['avg_latency_ms']:>15.1f} {dual_agent['avg_latency_ms']:>15.1f}")
    print(f"  {'P50 Latency (ms)':<30} {traditional['p50_ms']:>15.1f} {dual_agent['p50_ms']:>15.1f}")
    print(f"  {'P70 Latency (ms)':<30} {traditional['p70_ms']:>15.1f} {dual_agent['p70_ms']:>15.1f}")
    print(f"  {'P100 Latency (ms)':<30} {traditional['p100_ms']:>15.1f} {dual_agent['p100_ms']:>15.1f}")
    print(f"  {'Cache Hit Rate':<30} {'N/A':>15} {dual_agent['cache_hit_rate']:>15}")

    speedup = traditional['avg_latency_ms'] / dual_agent['avg_latency_ms'] if dual_agent['avg_latency_ms'] > 0 else 0
    print(f"\n  Speedup: {speedup:.2f}x")

    if args.output:
        results = {
            "traditional": traditional,
            "dual_agent": dual_agent,
            "speedup": round(speedup, 2),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
