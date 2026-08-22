"""Grid-search hyperparameter tuner for QA & Guardrail thresholds."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.config import QUERIES_PATH


async def evaluate_combo(
    queries: list[dict],
    null_thresh: float,
    qa_margin: float,
    rel_margin: float,
) -> dict:
    """Evaluate pipeline accuracy under specific threshold combo."""
    import app.config as cfg
    from app.harness.orchestrator import run_pipeline
    from app.harness.query_cache import clear_cache

    cfg.QA_NULL_THRESHOLD = null_thresh
    cfg.QA_MARGIN_THRESHOLD = qa_margin
    cfg.RELEVANCE_MARGIN_THRESHOLD = rel_margin

    clear_cache()

    correct_answers = 0
    total_evaluated = 0
    total_latency_ms = 0.0

    for q_item in queries[:40]:  # Evaluate subset for fast tuning
        q_text = q_item.get("query", "").strip()
        gold_ans = q_item.get("gold_answer", "").strip().lower()
        if not q_text:
            continue

        total_evaluated += 1
        res = await run_pipeline(q_text)
        total_latency_ms += res.metrics.total_ms

        pred_ans = (res.answer or "").strip().lower()
        # Check string match / token overlap with gold answer
        if gold_ans and (gold_ans in pred_ans or pred_ans in gold_ans):
            correct_answers += 1
        elif res.relevant:
            correct_answers += 0.5

    acc = (correct_answers / max(1, total_evaluated)) * 100
    avg_lat = total_latency_ms / max(1, total_evaluated)

    return {
        "null_thresh": null_thresh,
        "qa_margin": qa_margin,
        "rel_margin": rel_margin,
        "accuracy_pct": round(acc, 2),
        "avg_latency_ms": round(avg_lat, 2),
    }


async def main():
    print("======================================================================")
    print("HYPERPARAMETER THRESHOLD GRID SEARCH")
    print("======================================================================")

    if not Path(QUERIES_PATH).exists():
        print(f"Error: {QUERIES_PATH} not found.")
        return

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} benchmark queries for tuning.\n")

    null_grid = [-0.5, 0.0, 0.5]
    qa_margin_grid = [0.0, 0.05, 0.1]
    rel_margin_grid = [0.0, 0.05, 0.1]

    best_score = -1.0
    best_combo = None
    results = []

    for n in null_grid:
        for qm in qa_margin_grid:
            for rm in rel_margin_grid:
                res = await evaluate_combo(queries, n, qm, rm)
                results.append(res)
                print(
                    f"Null: {n:<5} | QA Margin: {qm:<5} | Rel Margin: {rm:<5} "
                    f"-> Acc: {res['accuracy_pct']:>6.2f}% | Latency: {res['avg_latency_ms']:>6.2f} ms"
                )
                if res["accuracy_pct"] > best_score:
                    best_score = res["accuracy_pct"]
                    best_combo = res

    print("\n======================================================================")
    print("OPTIMAL THRESHOLD CONFIGURATION FOUND:")
    print("======================================================================")
    print(json.dumps(best_combo, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
