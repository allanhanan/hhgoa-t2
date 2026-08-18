"""Download and sample MSMARCO-XI dataset for index building.

Usage:
    python -m ingestion.download_dataset [--n-passages 500000] [--output-dir data/]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


TRAIN_FILES = [
    "train/hintrain.parquet",
]


def download_and_extract(
    n_passages: int = 8_800_000,
    output_dir: str = "data",
    language_files: list[str] | str | None = None,
):
    """Download MSMARCO-XI parquet files and extract unique passages and benchmark queries.

    Streams extracted passages directly to disk using PyArrow ParquetWriter to maintain low RAM (<200MB).
    """
    from huggingface_hub import hf_hub_download
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    passages_path = output_path / "passages.parquet"
    queries_path = output_path / "benchmark_queries.json"

    schema = pa.schema([
        ("id", pa.int64()),
        ("text", pa.string()),
        ("is_selected", pa.int8()),
        ("query_type", pa.string()),
        ("source_query", pa.string()),
    ])

    writer = pq.ParquetWriter(passages_path, schema, compression="snappy")

    if isinstance(language_files, str):
        files = [language_files]
    elif isinstance(language_files, list):
        files = language_files
    else:
        files = TRAIN_FILES

    passages_batch: list[dict] = []
    queries: list[dict] = []
    seen_hashes: set[str] = set()
    total_passages = 0
    row_count = 0

    try:
        for lfile in files:
            if total_passages >= n_passages:
                break

            logger.info(f"Downloading/loading MSMARCO-XI file ({lfile}) via HF Hub...")
            try:
                local_file_path = hf_hub_download(
                    repo_id="ai4bharat/MSMARCO-XI",
                    filename=lfile,
                    repo_type="dataset",
                )
            except Exception as e:
                logger.warning(f"Failed to download {lfile}: {e}")
                continue

            local_file = Path(local_file_path)
            logger.info(f"Parquet file ready at {local_file} ({local_file.stat().st_size / (1024*1024):.1f} MB)")

            pf = pq.ParquetFile(str(local_file))
            logger.info(f"Opened {lfile}: {pf.metadata.num_rows} rows")

            for batch in pf.iter_batches(batch_size=4096, columns=["query", "Eng_Query", "Answer", "Eng_Answer", "query_type", "query_id", "passages"]):
                b_dict = batch.to_pydict()
                queries_col = b_dict.get("Eng_Query") or b_dict.get("query") or []
                answers_col = b_dict.get("Eng_Answer") or b_dict.get("Answer") or []
                qtypes = b_dict.get("query_type") or []
                qids = b_dict.get("query_id") or []
                passages_list = b_dict.get("passages") or []

                for i in range(len(queries_col)):
                    row_count += 1
                    eq = queries_col[i]
                    ea = answers_col[i] if i < len(answers_col) else ""

                    # Keep only top 500 queries for benchmark to save RAM
                    if eq and ea and len(queries) < 500:
                        queries.append({
                            "query": str(b_dict["query"][i] if "query" in b_dict else ""),
                            "eng_query": str(eq),
                            "answer": str(b_dict["Answer"][i] if "Answer" in b_dict else ""),
                            "eng_answer": str(ea),
                            "query_type": str(qtypes[i] if i < len(qtypes) else ""),
                            "query_id": str(qids[i] if i < len(qids) else ""),
                        })

                    p_struct = passages_list[i] if i < len(passages_list) else {}
                    if isinstance(p_struct, dict):
                        eng_passages = p_struct.get("English_passages") or p_struct.get("passage_text") or []
                        is_selected = p_struct.get("is_selected") or []

                        for j, ptext in enumerate(eng_passages):
                            if not ptext:
                                continue
                            ptext_str = str(ptext).strip()
                            if not ptext_str:
                                continue

                            h = hashlib.md5(ptext_str.encode()).hexdigest()
                            if h in seen_hashes:
                                continue
                            seen_hashes.add(h)

                            sel = is_selected[j] if j < len(is_selected) else 0

                            passages_batch.append({
                                "id": total_passages,
                                "text": ptext_str,
                                "is_selected": int(sel),
                                "query_type": str(qtypes[i] if i < len(qtypes) else ""),
                                "source_query": str(eq or ""),
                            })
                            total_passages += 1

                            # Flush batch to disk every 50,000 passages to keep RAM ultra-low (<100MB)
                            if len(passages_batch) >= 50_000:
                                table = pa.table({
                                    "id": [p["id"] for p in passages_batch],
                                    "text": [p["text"] for p in passages_batch],
                                    "is_selected": [p["is_selected"] for p in passages_batch],
                                    "query_type": [p["query_type"] for p in passages_batch],
                                    "source_query": [p["source_query"] for p in passages_batch],
                                }, schema=schema)
                                writer.write_table(table)
                                passages_batch.clear()

                            if total_passages >= n_passages:
                                break

                    if total_passages >= n_passages:
                        break

                if total_passages >= n_passages:
                    break

                if row_count % 50000 == 0:
                    logger.info(f"  Processed {row_count} rows, {total_passages} unique passages (RAM buffered: {len(passages_batch)})")

        # Flush remaining buffer
        if passages_batch:
            table = pa.table({
                "id": [p["id"] for p in passages_batch],
                "text": [p["text"] for p in passages_batch],
                "is_selected": [p["is_selected"] for p in passages_batch],
                "query_type": [p["query_type"] for p in passages_batch],
                "source_query": [p["source_query"] for p in passages_batch],
            }, schema=schema)
            writer.write_table(table)
            passages_batch.clear()

    finally:
        writer.close()

    logger.info(f"Final: {total_passages} passages written to disk from {row_count} rows")

    # Save benchmark queries (max 200)
    benchmark_queries = [q for q in queries if q["answer"]][:200]
    with open(queries_path, "w") as f:
        json.dump(benchmark_queries, f, indent=2)
    logger.info(f"Saved {len(benchmark_queries)} benchmark queries to {queries_path}")

    return passages_path, queries_path


def main():
    parser = argparse.ArgumentParser(description="Download and sample MSMARCO-XI")
    parser.add_argument("--n-passages", type=int, default=500_000)
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    download_and_extract(
        n_passages=args.n_passages,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
