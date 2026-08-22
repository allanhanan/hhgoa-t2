"""Hyperparameter tuning and latency landscape graphing suite.

Sweeps across RAG pipeline thresholds/hyperparameters and produces:
1. Terminal ASCII latency response curves (Gradient descent / optimization curves: Y = Latency, X = Hyperparam)
2. High-resolution Matplotlib plot ('threshold_tuning_curves.png') with minimum latency annotations.

Usage:
    python tune_thresholds.py [--queries 30] [--plot-output threshold_tuning_curves.png]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from app.config import QUERIES_PATH


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def render_ascii_graph(
    x_vals: list[float | int],
    y_vals: list[float],
    title: str,
    x_label: str,
    y_label: str = "Latency (ms)",
    height: int = 10,
    width: int = 40,
) -> str:
    """Render a 2D terminal ASCII curve of Y vs X with minimum point marked."""
    if not y_vals or len(y_vals) < 2:
        return ""

    min_y = min(y_vals)
    max_y = max(y_vals)
    y_range = max(max_y - min_y, 1e-6)

    min_idx = y_vals.index(min_y)

    lines = []
    lines.append(f"\n+-- {title.upper()} " + "-" * max(10, 45 - len(title)) + "+")

    # Grid rows from top (max latency) to bottom (min latency)
    for r in range(height, -1, -1):
        val_at_row = min_y + (r / height) * y_range
        row_str = f"{val_at_row:>6.1f} | "

        # For each data point, determine its column and row
        row_chars = [" "] * len(x_vals)
        for i, y in enumerate(y_vals):
            # Calculate corresponding row
            y_row = round(((y - min_y) / y_range) * height)
            if y_row == r:
                if i == min_idx:
                    row_chars[i] = "*"  # Minimum latency highlight
                else:
                    row_chars[i] = "o"
            elif y_row > r and (i > 0 and round(((y_vals[i - 1] - min_y) / y_range) * height) < r):
                row_chars[i] = "|"

        # Interpolate visually across width
        formatted_row = "   ".join(row_chars)
        lines.append(row_str + formatted_row)

    # X-axis line
    axis_line = "       +" + "-" * (len(x_vals) * 4 + 2)
    lines.append(axis_line)

    # X labels
    x_ticks = "         " + "  ".join(f"{x:>4}" for x in x_vals)
    lines.append(x_ticks)
    lines.append(f"         ^ {x_label} (* = Lowest Latency: {min_y:.2f} ms at {x_vals[min_idx]})")
    lines.append("+" + "-" * 58 + "+")

    return "\n".join(lines)


async def evaluate_config(
    queries: list[dict],
    null_thresh: float,
    qa_margin: float,
    rel_margin: float,
    ann_top_k: int = 8,
    min_abs_score: float = 0.45,
    max_queries: int = 25,
) -> dict[str, float]:
    """Evaluate pipeline accuracy and latency under a specific threshold configuration."""
    import app.config as cfg
    from app.harness.orchestrator import run_pipeline
    from app.harness.query_cache import clear_cache

    cfg.QA_NULL_THRESHOLD = null_thresh
    cfg.QA_MARGIN_THRESHOLD = qa_margin
    cfg.RELEVANCE_MARGIN_THRESHOLD = rel_margin
    cfg.ANN_TOP_K = ann_top_k
    cfg.RELEVANCE_MIN_ABS_SCORE = min_abs_score

    clear_cache()

    correct_answers = 0
    total_evaluated = 0
    latencies = []

    eval_subset = queries[:max_queries]
    for q_item in eval_subset:
        q_text = q_item.get("query", "").strip() or q_item.get("eng_query", "").strip()
        gold_ans = q_item.get("gold_answer", "").strip().lower()
        if not q_text:
            continue

        total_evaluated += 1
        t0 = time.perf_counter()
        res = await run_pipeline(q_text)
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)

        pred_ans = (res.answer or "").strip().lower()
        if gold_ans and (gold_ans in pred_ans or pred_ans in gold_ans):
            correct_answers += 1
        elif res.relevant and res.answer:
            correct_answers += 0.6

    acc = (correct_answers / max(1, total_evaluated)) * 100
    avg_lat = sum(latencies) / max(1, len(latencies))

    return {
        "null_thresh": null_thresh,
        "qa_margin": qa_margin,
        "rel_margin": rel_margin,
        "ann_top_k": ann_top_k,
        "min_abs_score": min_abs_score,
        "accuracy_pct": round(acc, 2),
        "avg_latency_ms": round(avg_lat, 2),
    }


async def run_hyperparameter_sweeps(queries: list[dict], max_queries: int = 25):
    """Run individual and gradient sweeps across key hyperparameters."""
    sweep_results: dict[str, list[dict]] = {}

    print("\n" + "=" * 70)
    print(" >>> RUNNING HYPERPARAMETER LATENCY SWEEPS")
    print("=" * 70)

    # 1. Sweep QA_NULL_THRESHOLD (controls span rejection vs fast exit)
    print("\n[1/4] Sweeping QA_NULL_THRESHOLD [-0.5 -> 1.0]...")
    null_vals = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    null_data = []
    for val in null_vals:
        res = await evaluate_config(queries, null_thresh=val, qa_margin=0.05, rel_margin=0.0, max_queries=max_queries)
        null_data.append(res)
        print(f"  QA_NULL_THRESHOLD = {val:>5.2f} -> Latency: {res['avg_latency_ms']:>6.2f} ms | Acc: {res['accuracy_pct']:>5.1f}%")
    sweep_results["QA_NULL_THRESHOLD"] = null_data

    # 2. Sweep QA_MARGIN_THRESHOLD
    print("\n[2/4] Sweeping QA_MARGIN_THRESHOLD [0.00 -> 0.20]...")
    margin_vals = [0.00, 0.03, 0.05, 0.08, 0.12, 0.20]
    margin_data = []
    for val in margin_vals:
        res = await evaluate_config(queries, null_thresh=0.0, qa_margin=val, rel_margin=0.0, max_queries=max_queries)
        margin_data.append(res)
        print(f"  QA_MARGIN_THRESHOLD = {val:>5.2f} -> Latency: {res['avg_latency_ms']:>6.2f} ms | Acc: {res['accuracy_pct']:>5.1f}%")
    sweep_results["QA_MARGIN_THRESHOLD"] = margin_data

    # 3. Sweep ANN_TOP_K (retrieval candidate pool size for rescoring)
    print("\n[3/4] Sweeping ANN_TOP_K [4 -> 64]...")
    k_vals = [4, 8, 16, 32, 64]
    k_data = []
    for val in k_vals:
        res = await evaluate_config(queries, null_thresh=0.0, qa_margin=0.05, rel_margin=0.0, ann_top_k=val, max_queries=max_queries)
        k_data.append(res)
        print(f"  ANN_TOP_K = {val:>3} -> Latency: {res['avg_latency_ms']:>6.2f} ms | Acc: {res['accuracy_pct']:>5.1f}%")
    sweep_results["ANN_TOP_K"] = k_data

    # 4. Sweep RELEVANCE_MIN_ABS_SCORE
    print("\n[4/4] Sweeping RELEVANCE_MIN_ABS_SCORE [0.35 -> 0.55]...")
    rel_vals = [0.35, 0.40, 0.45, 0.50, 0.55]
    rel_data = []
    for val in rel_vals:
        res = await evaluate_config(queries, null_thresh=0.0, qa_margin=0.05, rel_margin=0.0, min_abs_score=val, max_queries=max_queries)
        rel_data.append(res)
        print(f"  RELEVANCE_MIN_ABS_SCORE = {val:>5.2f} -> Latency: {res['avg_latency_ms']:>6.2f} ms | Acc: {res['accuracy_pct']:>5.1f}%")
    sweep_results["RELEVANCE_MIN_ABS_SCORE"] = rel_data

    return sweep_results


def save_matplotlib_graph(sweep_results: dict[str, list[dict]], output_path: str = "threshold_tuning_curves.png"):
    """Generate high-resolution gradient descent / response curve plots."""
    try:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("RAG Pipeline Hyperparameter Latency Landscapes (Gradient Curves)", fontsize=16, fontweight="bold", y=0.98)

        param_configs = [
            ("QA_NULL_THRESHOLD", "null_thresh", axes[0, 0], "#6366f1", "Null Threshold"),
            ("QA_MARGIN_THRESHOLD", "qa_margin", axes[0, 1], "#06b6d4", "QA Margin Threshold"),
            ("ANN_TOP_K", "ann_top_k", axes[1, 0], "#10b981", "ANN Top-K Candidates"),
            ("RELEVANCE_MIN_ABS_SCORE", "min_abs_score", axes[1, 1], "#f59e0b", "Relevance Min Abs Score"),
        ]

        for sweep_key, param_field, ax, color, display_name in param_configs:
            data = sweep_results.get(sweep_key, [])
            if not data:
                continue

            x_pts = [d[param_field] for d in data]
            y_pts = [d["avg_latency_ms"] for d in data]
            acc_pts = [d["accuracy_pct"] for d in data]

            # Line plot with markers
            ax.plot(x_pts, y_pts, color=color, linewidth=2.5, marker="o", markersize=8, label="Latency (ms)")

            # Find minimum latency point
            min_lat = min(y_pts)
            min_idx = y_pts.index(min_lat)
            opt_x = x_pts[min_idx]

            # Highlight minimum basin
            ax.scatter([opt_x], [min_lat], color="#ef4444", s=180, zorder=5, edgecolors="#ffffff", linewidth=2, label=f"Lowest Latency: {min_lat:.1f}ms")
            ax.annotate(
                f"Min: {min_lat:.1f} ms\n(Acc: {acc_pts[min_idx]:.1f}%)",
                (opt_x, min_lat),
                textcoords="offset points",
                xytext=(0, 14),
                ha="center",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec=color, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            )

            ax.set_title(f"Latency vs {display_name}", fontsize=12, fontweight="semibold")
            ax.set_xlabel(display_name, fontsize=10)
            ax.set_ylabel("Latency (ms) — Lower is Better", fontsize=10)
            ax.legend(loc="upper right", framealpha=0.9)
            ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"\n[OK] Visual gradient plots saved successfully to: {output_path}")

    except Exception as e:
        print(f"Note: Matplotlib plot generation skipped ({e})")


async def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Latency Curve Tuning Suite")
    parser.add_argument("--queries", type=int, default=25, help="Number of benchmark queries to evaluate per configuration")
    parser.add_argument("--plot-output", type=str, default="threshold_tuning_curves.png", help="Path to save visual latency plot")
    args = parser.parse_args()

    queries_file = Path(QUERIES_PATH)
    if not queries_file.exists():
        # Fallback queries if benchmark json is not found
        queries = [
            {"query": "What is retrieval augmented generation?", "gold_answer": "combines retrieval with generation"},
            {"query": "How does FAISS binary quantization work?", "gold_answer": "thresholded at zero"},
            {"query": "Who invented the telephone?", "gold_answer": "Alexander Graham Bell"},
            {"query": "What is machine learning?", "gold_answer": "algorithms that learn from data"},
            {"query": "What is natural language processing?", "gold_answer": "processing human language"},
        ] * 5
    else:
        with open(queries_file, "r", encoding="utf-8") as f:
            queries = json.load(f)

    print(f"Loaded {len(queries)} queries. Running tuning sweeps (evaluating {args.queries} queries per step)...")

    # Run sweeps
    sweep_results = await run_hyperparameter_sweeps(queries, max_queries=args.queries)

    # 1. Print ASCII terminal gradient descent graphs
    print("\n" + "=" * 70)
    print(" [LATENCY OPTIMIZATION CURVES] (Y = Latency ms, X = Hyperparam)")
    print("=" * 70)

    # QA_NULL_THRESHOLD graph
    null_data = sweep_results["QA_NULL_THRESHOLD"]
    print(render_ascii_graph(
        x_vals=[d["null_thresh"] for d in null_data],
        y_vals=[d["avg_latency_ms"] for d in null_data],
        title="Latency Curve vs QA_NULL_THRESHOLD",
        x_label="QA_NULL_THRESHOLD",
    ))

    # ANN_TOP_K graph
    k_data = sweep_results["ANN_TOP_K"]
    print(render_ascii_graph(
        x_vals=[d["ann_top_k"] for d in k_data],
        y_vals=[d["avg_latency_ms"] for d in k_data],
        title="Latency Curve vs ANN_TOP_K Candidates",
        x_label="ANN_TOP_K (Candidates Rescored)",
    ))

    # 2. Save high-res PNG plots
    save_matplotlib_graph(sweep_results, output_path=args.plot_output)

    # 3. Print optimal configuration summary
    print("\n" + "=" * 70)
    print(" [RECOMMENDED LOWEST LATENCY CONFIGURATION]")
    print("=" * 70)

    opt_null = min(sweep_results["QA_NULL_THRESHOLD"], key=lambda x: x["avg_latency_ms"])
    opt_k = min(sweep_results["ANN_TOP_K"], key=lambda x: x["avg_latency_ms"])
    opt_margin = min(sweep_results["QA_MARGIN_THRESHOLD"], key=lambda x: x["avg_latency_ms"])
    opt_rel = min(sweep_results["RELEVANCE_MIN_ABS_SCORE"], key=lambda x: x["avg_latency_ms"])

    print(f"  * QA_NULL_THRESHOLD:         {opt_null['null_thresh']}  (Latency: {opt_null['avg_latency_ms']} ms)")
    print(f"  * QA_MARGIN_THRESHOLD:       {opt_margin['qa_margin']}  (Latency: {opt_margin['avg_latency_ms']} ms)")
    print(f"  * ANN_TOP_K:                 {opt_k['ann_top_k']}  (Latency: {opt_k['avg_latency_ms']} ms)")
    print(f"  * RELEVANCE_MIN_ABS_SCORE:   {opt_rel['min_abs_score']}  (Latency: {opt_rel['avg_latency_ms']} ms)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

