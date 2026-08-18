"""RAG Pipeline Benchmark Suite.

Measures P50, P70, P100 latencies for:
- Retrieval only: embed + binary search + rescore + payload fetch
- Full pipeline: retrieval + LLM generation (TTFT)

Usage:
    python benchmark.py --mode retrieval --queries 100
    python benchmark.py --mode pipeline --queries 50
    python benchmark.py --mode all --queries 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from app.config import LATENCY_BUDGET_MS, RETRIEVAL_BUDGET_MS, DATA_DIR


def percentile(values: list[float], pct: float) -> float:
    """Compute percentile from a sorted list."""
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def load_benchmark_queries(path: str | None = None, n: int = 100) -> list[str]:
    """Load benchmark queries from the extracted dataset."""
    path = path or str(DATA_DIR / "benchmark_queries.json")
    try:
        with open(path) as f:
            queries = json.load(f)
        return [q["eng_query"] or q["query"] for q in queries[:n]]
    except FileNotFoundError:
        # Fallback queries
        return [
            "What is retrieval augmented generation?",
            "How does FAISS indexing work?",
            "What is machine learning?",
            "How do search engines rank results?",
            "What is natural language processing?",
            "How does binary quantization work?",
            "What is the purpose of embedding models?",
            "How do neural networks learn?",
        ]


def run_retrieval_benchmark(queries: list[str], n_runs: int = 100):
    """Benchmark retrieval-only latency (embed + search + rescore + payload)."""
    from app.embedder.encoder import encode_and_binarize, warmup
    from app.retriever.vector_db import load_index, search
    from app.retriever.rescorer import load_vectors, rescore
    from app.retriever.payload_store import connect, fetch

    print("Warming up...")
    warmup()
    load_index()
    load_vectors()
    connect()

    embed_ms_list = []
    search_ms_list = []
    rescore_ms_list = []
    payload_ms_list = []
    total_ms_list = []

    print(f"Running {n_runs} retrieval queries...")
    for i in range(n_runs):
        query = queries[i % len(queries)]
        t_total = time.perf_counter()

        # Embed
        t0 = time.perf_counter()
        float_emb, binary_emb = encode_and_binarize(query)
        embed_ms = (time.perf_counter() - t0) * 1000

        # Binary search
        t0 = time.perf_counter()
        distances, ids = search(binary_emb, top_k=100)
        search_ms = (time.perf_counter() - t0) * 1000

        # Rescore
        t0 = time.perf_counter()
        scored = rescore(float_emb, ids, top_k=5)
        rescore_ms = (time.perf_counter() - t0) * 1000

        # Payload fetch
        t0 = time.perf_counter()
        scored_ids = [s[0] for s in scored]
        passages = fetch(scored_ids)
        payload_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000

        embed_ms_list.append(embed_ms)
        search_ms_list.append(search_ms)
        rescore_ms_list.append(rescore_ms)
        payload_ms_list.append(payload_ms)
        total_ms_list.append(total_ms)

    # Print results
    print(f"\n{'='*70}")
    print(f"RETRIEVAL BENCHMARK ({n_runs} queries)")
    print(f"{'='*70}")
    print(f"{'stage':<15}{'avg':>8}{'P50':>8}{'P70':>8}{'P95':>8}{'P100':>8}   (ms)")
    print(f"{'-'*63}")

    results = {}
    for name, values in [
        ("embed", embed_ms_list),
        ("binary_search", search_ms_list),
        ("rescore", rescore_ms_list),
        ("payload", payload_ms_list),
        ("TOTAL", total_ms_list),
    ]:
        p50 = percentile(values, 50)
        p70 = percentile(values, 70)
        p95 = percentile(values, 95)
        p100 = percentile(values, 100)
        avg = statistics.mean(values)
        results[name] = {
            "avg": round(avg, 2),
            "p50": round(p50, 2),
            "p70": round(p70, 2),
            "p95": round(p95, 2),
            "p100": round(p100, 2),
        }
        print(
            f"{name:<15}"
            f"{avg:>8.2f}"
            f"{p50:>8.2f}"
            f"{p70:>8.2f}"
            f"{p95:>8.2f}"
            f"{p100:>8.2f}"
        )

    p100_total = percentile(total_ms_list, 100)
    p95_total = percentile(total_ms_list, 95)
    p50_total = percentile(total_ms_list, 50)
    print(f"\nBudget: {RETRIEVAL_BUDGET_MS}ms | P50: {p50_total:.2f}ms | P95: {p95_total:.2f}ms | P100: {p100_total:.2f}ms")
    if p100_total <= RETRIEVAL_BUDGET_MS:
        print("[PASS]: within retrieval budget")
    else:
        print("[FAIL]: over retrieval budget")

    return results


async def run_pipeline_benchmark(queries: list[str], n_runs: int = 50):
    """Benchmark full pipeline latency (retrieval + generation)."""
    from app.retriever.vector_db import load_index
    from app.retriever.rescorer import load_vectors
    from app.retriever.payload_store import connect
    from app.embedder.encoder import warmup
    from app.harness.orchestrator import run_pipeline

    print("Warming up...")
    warmup()
    load_index()
    load_vectors()
    connect()

    total_ms_list = []
    retrieval_ms_list = []
    ttft_ms_list = []

    print(f"Running {n_runs} pipeline queries...")
    for i in range(n_runs):
        query = queries[i % len(queries)]
        result = await run_pipeline(query)

        total_ms_list.append(result.metrics.total_ms)
        retrieval_ms_list.append(result.metrics.retrieval_ms)
        if result.metrics.generate_ttft_ms > 0:
            ttft_ms_list.append(result.metrics.generate_ttft_ms)

    # Print results
    print(f"\n{'='*70}")
    print(f"PIPELINE BENCHMARK ({n_runs} queries)")
    print(f"{'='*70}")
    print(f"{'stage':<15}{'avg':>8}{'P50':>8}{'P70':>8}{'P95':>8}{'P100':>8}   (ms)")
    print(f"{'-'*63}")

    results = {}
    for name, values in [
        ("retrieval", retrieval_ms_list),
        ("llm_ttft", ttft_ms_list or [0.0]),
        ("TOTAL", total_ms_list),
    ]:
        if values:
            p50 = percentile(values, 50)
            p70 = percentile(values, 70)
            p95 = percentile(values, 95)
            p100 = percentile(values, 100)
            avg = statistics.mean(values)
            results[name] = {
                "avg": round(avg, 2),
                "p50": round(p50, 2),
                "p70": round(p70, 2),
                "p95": round(p95, 2),
                "p100": round(p100, 2),
            }
            print(
                f"{name:<15}"
                f"{avg:>8.2f}"
                f"{p50:>8.2f}"
                f"{p70:>8.2f}"
                f"{p95:>8.2f}"
                f"{p100:>8.2f}"
            )

    p100_total = percentile(total_ms_list, 100)
    p95_total = percentile(total_ms_list, 95)
    p50_total = percentile(total_ms_list, 50)
    print(f"\nBudget: {LATENCY_BUDGET_MS}ms | P50: {p50_total:.2f}ms | P95: {p95_total:.2f}ms | P100: {p100_total:.2f}ms")
    if p100_total <= LATENCY_BUDGET_MS:
        print("[PASS]: within pipeline budget")
    else:
        print("[FAIL]: over pipeline budget")

    return results


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Benchmark")
    parser.add_argument("--mode", choices=["retrieval", "pipeline", "all"], default="retrieval")
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()

    queries = load_benchmark_queries(n=args.queries)
    print(f"Loaded {len(queries)} benchmark queries")

    if args.mode in ("retrieval", "all"):
        run_retrieval_benchmark(queries, n_runs=args.queries)

    if args.mode in ("pipeline", "all"):
        asyncio.run(run_pipeline_benchmark(queries, n_runs=min(args.queries, 50)))


if __name__ == "__main__":
    main()
