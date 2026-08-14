#!/usr/bin/env python3
"""Interactive CLI demo of the Voice-Optimized RAG dual-agent system.

Shows real-time metrics: cache hit rate, latency breakdown, and prefetch activity.

Usage:
    # With documents in a directory:
    python examples/cli_demo.py --docs path/to/docs/

    # With Groq (default LLM) and local Sentence Transformer embeddings:
    VOR_LLM_API_KEY=sk-... python examples/cli_demo.py --docs path/to/docs/

    # With Ollama (local):
    python examples/cli_demo.py --provider ollama --model llama3.2 --docs path/to/docs/
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add parent dir to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_optimized_rag import MemoryRouter, VORConfig


def print_metrics(router: MemoryRouter) -> None:
    """Print current system metrics."""
    m = router.metrics
    summary = m.summary()
    hit_rate = summary.get("cache_hit_rate", "N/A")
    print(f"\n  Cache hit rate: {hit_rate}")
    print(f"  Cache size: {router.cache.size}")

    latency = summary.get("latency", {})
    if "fast_talker" in latency:
        ft = latency["fast_talker"]
        if "total_response" in ft:
            print(f"  Avg response latency: {ft['total_response']['avg_ms']:.1f}ms")
        if "cache_lookup" in ft:
            print(f"  Avg cache lookup: {ft['cache_lookup']['avg_ms']:.2f}ms")
        if "fallback_retrieval" in ft:
            print(f"  Fallback retrievals: {ft['fallback_retrieval']['count']}")

    counters = summary.get("counters", {})
    if "prefetch_operations" in counters:
        print(f"  Prefetch operations: {counters['prefetch_operations']}")
    if "predictions_made" in counters:
        print(f"  Predictions made: {counters['predictions_made']}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Voice-Optimized RAG CLI Demo")
    parser.add_argument("--docs", type=Path, help="Directory of documents to ingest")
    parser.add_argument("--provider", default="groq", choices=["openai", "anthropic", "ollama", "gemini", "groq"])
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--api-key", default=None, help="API key (or set VOR_LLM_API_KEY)")
    parser.add_argument(
        "--embedding-provider",
        default="sentence-transformers",
        choices=["openai", "ollama", "sentence-transformers"],
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    # Build config
    config_kwargs = {
        "llm_provider": args.provider,
        "embedding_provider": args.embedding_provider,
    }
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

    if args.embedding_provider == "sentence-transformers":
        config_kwargs["embedding_model"] = "all-MiniLM-L6-v2"
        config_kwargs["embedding_dimension"] = 384

    config = VORConfig(**config_kwargs)
    router = MemoryRouter(config)

    print("=" * 60)
    print("  Voice-Optimized RAG — CLI Demo")
    print("  Dual-Agent Architecture: Slow Thinker + Fast Talker")
    print("=" * 60)

    await router.start(log_level=args.log_level)

    # Ingest documents if provided
    if args.docs:
        if args.docs.is_dir():
            print(f"\nIngesting documents from {args.docs}...")
            count = await router.ingest_directory(args.docs)
            print(f"Ingested {count} chunks.")
            router.save_index()
        else:
            print(f"Warning: {args.docs} is not a directory")

    print("\nType your questions below. Commands: /metrics, /clear, /quit\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "/q"):
                break
            elif user_input.lower() == "/metrics":
                print_metrics(router)
                continue
            elif user_input.lower() == "/clear":
                await router.cache.clear()
                print("  Cache cleared.")
                continue

            # Query with timing
            start = time.perf_counter()
            print("Assistant: ", end="", flush=True)

            async for chunk in router.query_stream(user_input):
                print(chunk, end="", flush=True)

            elapsed = (time.perf_counter() - start) * 1000
            print(f"\n  [{elapsed:.0f}ms | cache: {router.metrics.cache_hit_rate:.0%}]")

    except KeyboardInterrupt:
        print("\n")
    finally:
        await router.stop()
        print("\nFinal metrics:")
        print_metrics(router)
        print()


if __name__ == "__main__":
    asyncio.run(main())
