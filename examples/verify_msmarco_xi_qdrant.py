#!/usr/bin/env python3
"""Verify the MSMARCO-XI development sample collection in Qdrant."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.retrieval.embeddings import create_embedding_provider
from voice_optimized_rag.retrieval.qdrant_store import QdrantVectorStore


def _vector_size(config) -> int:
    vectors = config.config.params.vectors
    if hasattr(vectors, "size"):
        return int(vectors.size)
    first = next(iter(vectors.values()))
    return int(first.size)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Qdrant MSMARCO-XI sample ingestion")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="msmarco_xi_test")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--query", default="what was the immediate impact of the success of the Manhattan Project?")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant_url, timeout=30)
    collections = [c.name for c in client.get_collections().collections]
    if args.collection not in collections:
        raise SystemExit(f"Missing collection: {args.collection}")

    info = client.get_collection(args.collection)
    size = _vector_size(info)
    if size != 384:
        raise SystemExit(f"Expected vector dimension 384, got {size}")
    if not info.points_count:
        raise SystemExit(f"Collection {args.collection} has no points")

    sample_points, _ = client.scroll(
        collection_name=args.collection,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not sample_points or not (sample_points[0].payload or {}).get("text"):
        raise SystemExit("No payload text found in indexed points")

    config = VORConfig(
        vector_store_provider="qdrant",
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.collection,
        embedding_provider="sentence-transformers",
        embedding_model=args.embedding_model,
        embedding_dimension=384,
    )
    embeddings = create_embedding_provider(config)
    store = QdrantVectorStore(
        dimension=embeddings.dimension,
        url=args.qdrant_url,
        collection_name=args.collection,
    )
    query_embedding = await embeddings.embed_single(args.query)
    results = store.search(query_embedding, top_k=args.top_k)
    if not results:
        raise SystemExit("Retrieval returned no passages")

    print(f"collection_exists=true")
    print(f"vector_dimension={size}")
    print(f"points_count={info.points_count}")
    print(f"payload_has_text=true")
    print(f"retrieval_results={len(results)}")
    for i, result in enumerate(results, 1):
        print(f"[{i}] score={result.score:.4f} query_id={result.metadata.get('query_id', '')} text={result.text[:180]}")


if __name__ == "__main__":
    asyncio.run(main())
