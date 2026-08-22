"""Evaluate FAISS IndexIVFScalarQuantizer (SQ8) accuracy (Recall@1/5/10, MRR@10) and latency (P50/95/99/100) vs BinaryFlat baseline.

Usage:
    python evaluate_ivf.py [--queries 100]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import faiss
import numpy as np

from app.config import DATA_DIR, FLOAT16_PATH, INDEX_PATH, IVF_INDEX_PATH
from app.embedder.encoder import encode_and_binarize, warmup
from app.retriever.rescorer import load_vectors, rescore


def percentile(values: list[float], pct: float) -> float:
    """Compute percentile value from a list of floats."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (pct / 100.0)
    f, c = int(k), min(int(k) + 1, len(sorted_v) - 1)
    if f == c:
        return sorted_v[f]
    return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])


def compute_mrr(gt_ids: list[int], pred_ids: list[int], k: int = 10) -> float:
    """Compute Mean Reciprocal Rank (MRR@k)."""
    gt_set = set(gt_ids[:1])  # Top-1 ground truth document
    for rank, p_id in enumerate(pred_ids[:k], start=1):
        if p_id in gt_set:
            return 1.0 / rank
    return 0.0


def compute_recall(gt_ids: list[int], pred_ids: list[int], k: int) -> float:
    """Compute Recall@k against ground truth top-k."""
    gt_k = set(gt_ids[:k])
    if not gt_k:
        return 0.0
    pred_k = set(pred_ids[:k])
    return len(gt_k.intersection(pred_k)) / float(len(gt_k))


def main():
    parser = argparse.ArgumentParser(description="Evaluate IndexIVFScalarQuantizer (SQ8) vs BinaryFlat RAG Index")
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()

    print("Initializing models and loading vectors...", flush=True)
    warmup()
    load_vectors(warm=False)

    # Load benchmark queries
    queries_file = Path(DATA_DIR / "benchmark_queries.json")
    with open(queries_file) as f:
        q_data = json.load(f)
    queries = [q["eng_query"] or q["query"] for q in q_data[: args.queries]]
    print(f"Loaded {len(queries)} benchmark queries for evaluation.", flush=True)

    # Step 1: Compute Ground Truth top-10 for each query using BinaryFlat + FP16 Exact Rescore
    print("\n[1/3] Computing Ground Truth baseline using BinaryFlat + FP16 rescore...", flush=True)
    binary_index = faiss.read_index_binary(INDEX_PATH)
    faiss.omp_set_num_threads(3)

    ground_truth = []
    baseline_latencies = []

    for q in queries:
        t0 = time.perf_counter()
        f_emb, b_emb = encode_and_binarize(q)

        # Binary search
        b_dist, b_ids = binary_index.search(b_emb.reshape(1, -1), 100)

        # FP16 rescore top-10
        rescored = rescore(f_emb, b_ids[0], top_k=10)
        gt_ids = [r[0] for r in rescored]
        elapsed = (time.perf_counter() - t0) * 1000

        ground_truth.append((f_emb, gt_ids))
        baseline_latencies.append(elapsed)

    print(f"Baseline (BinaryFlat) Latency: P50={percentile(baseline_latencies, 50):.2f}ms | P95={percentile(baseline_latencies, 95):.2f}ms | P100={percentile(baseline_latencies, 100):.2f}ms", flush=True)

    # Step 2: Load IVF Index
    ivf_path = Path(IVF_INDEX_PATH)
    if not ivf_path.exists():
        print(f"Error: IVF index file {ivf_path} does not exist. Run ingestion.build_ivf_index first.", flush=True)
        return

    print(f"\n[2/3] Loading IndexIVFScalarQuantizer (SQ8) Index from {ivf_path}...", flush=True)
    ivf_index = faiss.read_index(str(ivf_path))
    faiss.omp_set_num_threads(3)

    # Step 3: Parameter Sweep over nprobe and ANN candidate count
    print("\n[3/3] Running Parameter Sweep (nprobe x ANN_TOP_K)...", flush=True)
    print(f"\n{'='*100}", flush=True)
    print(f"{'nprobe':<8}{'ANN_k':<8}{'Rec@1':<10}{'Rec@5':<10}{'Rec@10':<10}{'MRR@10':<10}{'P50(ms)':<10}{'P95(ms)':<10}{'P99(ms)':<10}{'P100(ms)':<10}", flush=True)
    print(f"{'-'*100}", flush=True)

    nprobe_list = [1, 2, 4, 8]
    cand_k_list = [16, 32, 64]

    # Include BinaryFlat baseline row
    print(f"{'BASE':<8}{'100':<8}{'1.000':<10}{'1.000':<10}{'1.000':<10}{'1.000':<10}{percentile(baseline_latencies, 50):<10.2f}{percentile(baseline_latencies, 95):<10.2f}{percentile(baseline_latencies, 99):<10.2f}{percentile(baseline_latencies, 100):<10.2f}", flush=True)

    for nprobe in nprobe_list:
        ivf_index.nprobe = nprobe
        for ann_k in cand_k_list:
            rec1_list, rec5_list, rec10_list, mrr_list = [], [], [], []
            latencies = []

            for f_emb, gt_ids in ground_truth:
                t0 = time.perf_counter()

                # IVF search
                _, ivf_ids = ivf_index.search(f_emb.reshape(1, -1), ann_k)

                # FP16 rescore top-10
                rescored = rescore(f_emb, ivf_ids[0], top_k=10)
                pred_ids = [r[0] for r in rescored]
                elapsed = (time.perf_counter() - t0) * 1000

                latencies.append(elapsed)
                rec1_list.append(compute_recall(gt_ids, pred_ids, 1))
                rec5_list.append(compute_recall(gt_ids, pred_ids, 5))
                rec10_list.append(compute_recall(gt_ids, pred_ids, 10))
                mrr_list.append(compute_mrr(gt_ids, pred_ids, 10))

            r1 = statistics.mean(rec1_list)
            r5 = statistics.mean(rec5_list)
            r10 = statistics.mean(rec10_list)
            mrr = statistics.mean(mrr_list)
            p50 = percentile(latencies, 50)
            p95 = percentile(latencies, 95)
            p99 = percentile(latencies, 99)
            p100 = percentile(latencies, 100)

            print(f"{nprobe:<8}{ann_k:<8}{r1:<10.3f}{r5:<10.3f}{r10:<10.3f}{mrr:<10.3f}{p50:<10.2f}{p95:<10.2f}{p99:<10.2f}{p100:<10.2f}", flush=True)

    print(f"{'='*100}\n", flush=True)


if __name__ == "__main__":
    main()
