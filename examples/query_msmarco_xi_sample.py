#!/usr/bin/env python3
"""Query the MSMARCO-XI development sample in Qdrant."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_optimized_rag import MemoryRouter, VORConfig
from voice_optimized_rag.retrieval.embeddings import create_embedding_provider
from voice_optimized_rag.retrieval.qdrant_store import QdrantVectorStore


async def search(args: argparse.Namespace) -> None:
    config = VORConfig(
        vector_store_provider="qdrant",
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.collection,
        embedding_provider="sentence-transformers",
        embedding_model=args.embedding_model,
        embedding_dimension=384,
        guardrail_min_relevance_score=args.min_score,
    )
    embeddings = create_embedding_provider(config)
    store = QdrantVectorStore(
        dimension=embeddings.dimension,
        url=config.qdrant_url,
        collection_name=config.qdrant_collection,
    )

    embed_start = time.perf_counter()
    query_embedding = await embeddings.embed_single(args.query)
    embed_ms = (time.perf_counter() - embed_start) * 1000

    retrieval_start = time.perf_counter()
    results = store.search(query_embedding, top_k=args.top_k)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    print(f"collection={args.collection} points={store.size} embedding_ms={embed_ms:.1f} retrieval_ms={retrieval_ms:.1f}")
    for i, result in enumerate(results, 1):
        query_id = result.metadata.get("query_id", "")
        eng_query = result.metadata.get("Eng_Query", "")
        print(f"\n[{i}] score={result.score:.4f} query_id={query_id}")
        if eng_query:
            print(f"Eng_Query: {eng_query}")
        print(result.text[: args.preview_chars])


async def rag(args: argparse.Namespace) -> None:
    config_kwargs = {
        "vector_store_provider": "qdrant",
        "qdrant_url": args.qdrant_url,
        "qdrant_collection": args.collection,
        "embedding_provider": "sentence-transformers",
        "embedding_model": args.embedding_model,
        "embedding_dimension": 384,
        "llm_provider": args.provider,
        "llm_model": args.model,
        "guardrail_min_relevance_score": args.min_score,
        "fast_talker_max_context_chunks": args.top_k,
        "slow_thinker_rate_limit": 9999,
    }
    if args.api_key:
        config_kwargs["llm_api_key"] = args.api_key
    config = VORConfig(**config_kwargs)
    if args.provider == "groq":
        config.llm_base_url = "https://api.groq.com/openai/v1"

    router = MemoryRouter(config)
    await router.start(log_level=args.log_level)
    try:
        start = time.perf_counter()
        answer = await router.query(args.query)
        total_ms = (time.perf_counter() - start) * 1000
        print(answer)
        print(f"\ntotal_rag_ms={total_ms:.1f}")
    finally:
        await router.stop()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Search or run RAG against the MSMARCO-XI sample collection")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["search", "rag"], default="search")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="msmarco_xi_test")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--preview-chars", type=int, default=700)
    parser.add_argument("--provider", default="groq", choices=["openai", "anthropic", "ollama", "gemini", "groq"])
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    if args.mode == "search":
        await search(args)
    else:
        await rag(args)


if __name__ == "__main__":
    asyncio.run(main())
