#!/usr/bin/env python3
"""Benchmark candidate embedding models before changing production defaults."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_optimized_rag import VORConfig
from voice_optimized_rag.retrieval.embeddings import create_embedding_provider


DEFAULT_TEXTS = [
    "NovaCRM pricing plans and billing limits",
    "नोवासीआरएम में लॉगिन समस्या कैसे ठीक करें?",
    "NovaCRM API authentication and webhook setup",
    "ನೋವಾಸಿಆರ್ಎಂ ಬೆಲೆ ಯೋಜನೆಗಳ ಬಗ್ಗೆ ಮಾಹಿತಿ",
]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark embedding model latency")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = []
    texts = (DEFAULT_TEXTS * ((args.batch_size // len(DEFAULT_TEXTS)) + 1))[: args.batch_size]
    for model in args.models:
        config = VORConfig(
            embedding_provider="sentence-transformers",
            embedding_model=model,
            embedding_dimension=384,
        )
        provider = create_embedding_provider(config)
        await provider.embed(texts[:1])
        latencies = []
        for _ in range(args.rounds):
            start = time.perf_counter()
            await provider.embed(texts)
            latencies.append((time.perf_counter() - start) * 1000)
        values = sorted(latencies)
        results.append({
            "model": model,
            "dimension": provider.dimension,
            "batch_size": args.batch_size,
            "avg_ms": round(sum(values) / len(values), 2),
            "p50_ms": round(values[len(values) // 2], 2),
            "p100_ms": round(values[-1], 2),
        })

    print(json.dumps(results, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
