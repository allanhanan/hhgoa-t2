"""RAG Pipeline Benchmark Suite.

Measures P50, P70, P100 latencies for:
- Retrieval only: embedding + ann_search (or binary_search fallback) + rescore + payload fetch
- Full pipeline: retrieval + LLM generation (TTFT)

Usage:
    python benchmark.py --mode retrieval --queries 100
    python benchmark.py --mode pipeline --queries 50
    python benchmark.py --mode all --queries 100
"""
from __future__ import annotations
import gc
  
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
import psutil

from app.config import LATENCY_BUDGET_MS, RETRIEVAL_BUDGET_MS, DATA_DIR


def get_rss_mb() -> float:
    """Get current process Resident Set Size (RSS) in MB."""
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


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
    from app.embedder.encoder import encode, encode_and_binarize, warmup
    from app.retriever import vector_db
    from app.retriever.vector_db import load_index, search
    from app.retriever.rescorer import load_vectors, rescore
    from app.retriever.payload_store import connect, fetch

    print("Warming up...")
    warmup()
    load_index()
    load_vectors()
    connect()

    # Pre-run retrieval dummy queries to warm embedding/search runtime buffers
    dummy_q = "What is artificial intelligence?"
    f_emb = encode(dummy_q)
    q_v = f_emb if vector_db.is_ivf() else encode_and_binarize(dummy_q)[1]
    d_ids, d_dists = search(q_v, top_k=8)
    rescore(f_emb, d_ids, top_k=5)

    gc.collect()
    gc.freeze()


    query_records = []

    print(f"Running {n_runs} retrieval queries...")
    for i in range(n_runs):
        query = queries[i % len(queries)]
        rss_before = get_rss_mb()
        t_total = time.perf_counter()

        # Embed
        t0 = time.perf_counter()
        if vector_db.is_ivf():
            float_emb = encode(query)
            binary_emb = None
        else:
            float_emb, binary_emb = encode_and_binarize(query)
        embed_ms = (time.perf_counter() - t0) * 1000


        # Vector search
        t0 = time.perf_counter()
        from app.retriever import vector_db
        from app.config import ANN_TOP_K
        q_vec = float_emb if vector_db.is_ivf() else binary_emb
        k_search = ANN_TOP_K if vector_db.is_ivf() else 100
        distances, ids = search(q_vec, top_k=k_search)
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
        rss_after = get_rss_mb()

        rec = {
            "index": i,
            "query": query,
            "embed_ms": embed_ms,
            "ann_search_ms": search_ms,
            "rescore_ms": rescore_ms,
            "payload_ms": payload_ms,
            "total_ms": total_ms,
            "rss_mb": rss_after,
        }
        query_records.append(rec)

    embed_ms_list = [r["embed_ms"] for r in query_records]
    search_ms_list = [r["ann_search_ms"] for r in query_records]
    rescore_ms_list = [r["rescore_ms"] for r in query_records]
    payload_ms_list = [r["payload_ms"] for r in query_records]
    total_ms_list = [r["total_ms"] for r in query_records]

    # Print results
    print(f"\n{'='*70}")
    print(f"RETRIEVAL BENCHMARK ({n_runs} queries)")
    print(f"{'='*70}")
    print(f"{'stage':<25}{'avg':>8}{'P50':>8}{'P70':>8}{'P100':>8}   (ms)")
    print(f"{'-'*65}")

    from app.retriever import vector_db
    search_label = "ann_search" if vector_db.is_ivf() else "binary_search fallback"

    for name, values in [
        ("embedding", embed_ms_list),
        (search_label, search_ms_list),
        ("rescore", rescore_ms_list),
        ("payload", payload_ms_list),
        ("total retrieval", total_ms_list),
    ]:
        print(
            f"{name:<25}"
            f"{statistics.mean(values):>8.2f}"
            f"{percentile(values, 50):>8.2f}"
            f"{percentile(values, 70):>8.2f}"
            f"{percentile(values, 100):>8.2f}"
        )

    p100_total = percentile(total_ms_list, 100)
    p70_total = percentile(total_ms_list, 70)
    print(f"\nBudget: {RETRIEVAL_BUDGET_MS}ms | P70: {p70_total:.2f}ms | P100: {p100_total:.2f}ms")
    if p100_total <= RETRIEVAL_BUDGET_MS:
        print("[PASS] within retrieval budget")
    else:
        print("[FAIL] over retrieval budget")


    # Print slowest 10 queries
    print(f"\n{'='*95}")
    print("SLOWEST 10 QUERIES BY TOTAL RETRIEVAL TIME")
    print(f"{'='*95}")
    print(f"{'idx':<5}{'query':<30}{'embed(ms)':>10}{'ANN(ms)':>10}{'rescore(ms)':>12}{'payload(ms)':>12}{'total(ms)':>11}{'RSS(MB)':>10}")
    print(f"{'-'*95}")

    slowest_10 = sorted(query_records, key=lambda x: x["total_ms"], reverse=True)[:10]
    for q in slowest_10:
        q_str = (q["query"][:27] + "...") if len(q["query"]) > 30 else q["query"]
        print(
            f"{q['index']:<5}"
            f"{q_str:<30}"
            f"{q['embed_ms']:>10.2f}"
            f"{q['ann_search_ms']:>10.2f}"
            f"{q['rescore_ms']:>12.2f}"
            f"{q['payload_ms']:>12.2f}"
            f"{q['total_ms']:>11.2f}"
            f"{q['rss_mb']:>10.1f}"
        )

    return total_ms_list


async def run_pipeline_benchmark(queries: list[str], n_runs: int = 50):
    """Benchmark full pipeline latency (retrieval + generation)."""
    from app.retriever.vector_db import load_index
    from app.retriever.rescorer import load_vectors
    from app.retriever.payload_store import connect
    from app.embedder.encoder import warmup
    from app.answerer.extractive_qa import warmup as warmup_qa
    from app.harness.orchestrator import run_pipeline, warmup_pipeline

    print("Warming up index, models, and pipeline...")
    warmup()
    load_index()
    load_vectors()
    connect()
    warmup_qa()
    await warmup_pipeline()
    gc.collect()
    gc.freeze()





    query_records = []

    print(f"Running {n_runs} pipeline queries...")
    for i in range(n_runs):
        query = queries[i % len(queries)]
        rss_before = get_rss_mb()
        result = await run_pipeline(query)
        rss_after = get_rss_mb()

        total = result.metrics.total_ms
        retrieval = result.metrics.retrieval_ms
        answer = result.metrics.answer_extract_ms or result.metrics.answer_ms

        if answer > total + 1.0:
            print(f"⚠️ [WARNING] Query {i}: answer_extract_ms ({answer:.2f}ms) > total_ms ({total:.2f}ms) for query '{query[:30]}...'")

        rec = {
            "index": i,
            "query": query,
            "embed_ms": result.metrics.embed_ms,
            "ann_search_ms": result.metrics.search_ms,
            "rescore_ms": result.metrics.rescore_ms,
            "payload_ms": result.metrics.payload_ms,
            "retrieval_ms": retrieval,
            "answer_ms": answer,
            "total_ms": total,
            "rss_mb": rss_after,
        }
        query_records.append(rec)

    total_ms_list = [r["total_ms"] for r in query_records]
    retrieval_ms_list = [r["retrieval_ms"] for r in query_records]
    answer_ms_list = [r["answer_ms"] for r in query_records]

    # Print results
    print(f"\n{'='*70}")
    print(f"PIPELINE BENCHMARK ({n_runs} queries)")
    print(f"{'='*70}")
    print(f"{'stage':<15}{'avg':>8}{'P50':>8}{'P70':>8}{'P100':>8}   (ms)")
    print(f"{'-'*55}")

    for name, values in [
        ("retrieval", retrieval_ms_list),
        ("answer_extract", answer_ms_list),
        ("TOTAL", total_ms_list),
    ]:
        if values:
            print(
                f"{name:<15}"
                f"{statistics.mean(values):>8.2f}"
                f"{percentile(values, 50):>8.2f}"
                f"{percentile(values, 70):>8.2f}"
                f"{percentile(values, 100):>8.2f}"
            )

    p100_total = percentile(total_ms_list, 100)
    print(f"\nBudget: {LATENCY_BUDGET_MS}ms | P100: {p100_total:.2f}ms")
    if p100_total <= LATENCY_BUDGET_MS:
        print("[PASS] within pipeline budget")
    else:
        print("[FAIL] over pipeline budget")


    # Print slowest 10 queries
    print(f"\n{'='*95}")
    print("SLOWEST 10 PIPELINE QUERIES BY TOTAL LATENCY")
    print(f"{'='*95}")
    print(f"{'idx':<5}{'query':<30}{'embed(ms)':>10}{'ANN(ms)':>10}{'rescore(ms)':>12}{'payload(ms)':>12}{'total(ms)':>11}{'RSS(MB)':>10}")
    print(f"{'-'*95}")

    slowest_10 = sorted(query_records, key=lambda x: x["total_ms"], reverse=True)[:10]
    for q in slowest_10:
        q_str = (q["query"][:27] + "...") if len(q["query"]) > 30 else q["query"]
        print(
            f"{q['index']:<5}"
            f"{q_str:<30}"
            f"{q['embed_ms']:>10.2f}"
            f"{q['ann_search_ms']:>10.2f}"
            f"{q['rescore_ms']:>12.2f}"
            f"{q['payload_ms']:>12.2f}"
            f"{q['total_ms']:>11.2f}"
            f"{q['rss_mb']:>10.1f}"
        )

    return total_ms_list


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Benchmark")
    parser.add_argument("--mode", choices=["retrieval", "pipeline", "all"], default="retrieval")
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "all":
        import subprocess
        print("=== Running isolated retrieval benchmark ===")
        subprocess.run([sys.executable, __file__, "--mode", "retrieval", "--queries", str(args.queries)], check=True)
        print("\n=== Running isolated pipeline benchmark ===")
        subprocess.run([sys.executable, __file__, "--mode", "pipeline", "--queries", str(args.queries)], check=True)
        return

    queries = load_benchmark_queries(n=args.queries)
    print(f"Loaded {len(queries)} benchmark queries")

    if args.mode == "retrieval":
        run_retrieval_benchmark(queries, n_runs=args.queries)

    if args.mode == "pipeline":
        asyncio.run(run_pipeline_benchmark(queries, n_runs=min(args.queries, 50)))


if __name__ == "__main__":
    main()
