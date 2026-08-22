"""
Calibration script for RELEVANCE_MIN_ABS_SCORE / RELEVANCE_MARGIN_THRESHOLD.

READ-ONLY: does not modify app/config.py or any other file in your repo.
It just runs two batches of queries through your real retrieval pipeline
(encode -> vector search -> rescore) and prints the top-score / margin
numbers side by side, so you can see whether a single threshold can
actually tell "real" queries apart from "off-topic" ones -- instead of
picking a number blindly.

  IN-DOMAIN  = real questions sampled from data/benchmark_queries.json
               (things your corpus is known to have answers for)
  OFF-DOMAIN = a fixed list of questions about topics your corpus
               should NOT contain. Edit OFF_DOMAIN_QUERIES below if you
               know your corpus does cover some of these.

Usage:
    python calibrate_relevance.py
    python calibrate_relevance.py --n 100
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from app.config import QUERIES_PATH

# Edit this list if you want to test different off-topic questions.
OFF_DOMAIN_QUERIES = [
    "What is retrieval augmented generation?",
    "How do I train a large language model?",
    "What's the best way to learn the guitar?",
    "Explain quantum entanglement in simple terms.",
    "What's a good recipe for chocolate chip cookies?",
    "How does blockchain consensus work?",
    "What are the rules of cricket?",
    "Recommend a sci-fi novel from the 1980s.",
    "How do I fix a flat bicycle tire?",
    "What causes inflation in an economy?",
]


def load_in_domain_queries(n: int) -> list[str]:
    """Pull real queries straight from benchmark_queries.json (same file
    benchmark.py and tune_thresholds.py already use)."""
    path = Path(QUERIES_PATH)
    if not path.exists():
        print(f"[warn] {path} not found -- skipping in-domain batch.")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: list[str] = []
    for row in data:
        q = (row.get("eng_query") or row.get("query") or "").strip()
        if q:
            out.append(q)
        if len(out) >= n:
            break
    return out


def score_query(query: str) -> tuple[float, float]:
    """Run one query through encode -> search -> rescore, exactly like
    orchestrator.py does, and return (top_score, margin_over_rest)."""
    from app.embedder.encoder import encode, encode_and_binarize
    from app.retriever import vector_db
    from app.retriever.rescorer import rescore
    from app.config import TOP_K_BINARY, ANN_TOP_K

    if vector_db.is_ivf():
        float_emb = encode(query)
        q_vec = float_emb
        k_search = ANN_TOP_K
    else:
        float_emb, binary_emb = encode_and_binarize(query)
        q_vec = binary_emb
        k_search = TOP_K_BINARY

    _, ids = vector_db.search(q_vec, top_k=k_search)
    scored = rescore(float_emb, ids, top_k=5)

    if not scored:
        return 0.0, 0.0

    scores = sorted((s for _, s in scored), reverse=True)
    top = scores[0]
    rest_mean = sum(scores[1:]) / len(scores[1:]) if len(scores) > 1 else top
    margin = top - rest_mean
    return top, margin


def summarize(label: str, rows: list[tuple[str, float, float]]) -> None:
    if not rows:
        print(f"\n{label}: no queries scored.")
        return
    tops = [r[1] for r in rows]
    margins = [r[2] for r in rows]
    print(f"\n{label}  (n={len(rows)})")
    print("-" * 60)
    print(
        f"  top score   min={min(tops):.3f}  mean={statistics.mean(tops):.3f}  "
        f"median={statistics.median(tops):.3f}  max={max(tops):.3f}"
    )
    print(
        f"  margin      min={min(margins):.3f}  mean={statistics.mean(margins):.3f}  "
        f"median={statistics.median(margins):.3f}  max={max(margins):.3f}"
    )
    worst = sorted(rows, key=lambda r: r[1])[:5]
    best = sorted(rows, key=lambda r: r[1], reverse=True)[:5]
    print("  lowest top-score examples:")
    for q, t, m in worst:
        print(f"    {t:.3f}  (margin {m:.3f})  {q[:70]}")
    print("  highest top-score examples:")
    for q, t, m in best:
        print(f"    {t:.3f}  (margin {m:.3f})  {q[:70]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="number of in-domain queries to sample")
    args = ap.parse_args()

    from app.retriever.vector_db import load_index
    from app.retriever.rescorer import load_vectors
    from app.embedder.encoder import warmup

    print("Loading index + vectors...")
    warmup()
    load_index()
    load_vectors()

    in_domain_qs = load_in_domain_queries(args.n)
    off_domain_qs = OFF_DOMAIN_QUERIES

    in_rows = []
    for q in in_domain_qs:
        top, margin = score_query(q)
        in_rows.append((q, top, margin))

    off_rows = []
    for q in off_domain_qs:
        top, margin = score_query(q)
        off_rows.append((q, top, margin))

    summarize("IN-DOMAIN (pipeline SHOULD accept these)", in_rows)
    summarize("OFF-DOMAIN (pipeline SHOULD reject these)", off_rows)

    if in_rows and off_rows:
        in_tops = [r[1] for r in in_rows]
        off_tops = [r[1] for r in off_rows]
        gap_lo = min(in_tops)   # worst real query
        gap_hi = max(off_tops)  # best-scoring fake query

        print("\n" + "=" * 60)
        print("SUGGESTED CUTOFF")
        print("=" * 60)
        if gap_lo > gap_hi:
            suggested = (gap_lo + gap_hi) / 2
            print(f"Clean separation: off-domain max={gap_hi:.3f}, in-domain min={gap_lo:.3f}")
            print(f"-> RELEVANCE_MIN_ABS_SCORE ~= {suggested:.3f} would separate these two "
                  f"batches perfectly.")
        else:
            print(f"NO clean separation: off-domain max={gap_hi:.3f} overlaps with "
                  f"in-domain min={gap_lo:.3f}.")
            print("A single absolute-score cutoff can't fully separate these with the current "
                  "embedding model -- some real queries will always score similar to, or lower "
                  "than, some fake ones. Any threshold you pick is a trade-off between:")
            print("  - too low  -> off-topic questions still get answered (what you saw)")
            print("  - too high -> some real questions get wrongly rejected")
            print("If that trade-off isn't acceptable, the fix is a reranker (cross-encoder) "
                  "on the top candidates, not a better threshold number.")


if __name__ == "__main__":
    main()