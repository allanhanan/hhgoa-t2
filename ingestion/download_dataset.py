"""Download and sample MSMARCO-XI dataset for index building.

Usage:
    python -m ingestion.download_dataset [--n-passages 10000] [--output-dir data/]
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


import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

TRAIN_FILES = [
    "validation/hinval.parquet",
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

            local_file = output_path / Path(lfile).name
            if local_file.exists() and local_file.stat().st_size > 10 * 1024 * 1024:
                try:
                    pf_check = pq.ParquetFile(str(local_file))
                    logger.info(f"Using existing valid local file at {local_file} ({local_file.stat().st_size / (1024*1024):.1f} MB, {pf_check.metadata.num_rows} rows)")
                except Exception:
                    logger.warning(f"Existing local file {local_file} is incomplete/corrupted. Removing and re-downloading...")
                    try:
                        local_file.unlink()
                    except Exception as unlink_err:
                        logger.warning(f"Could not remove {local_file}: {unlink_err}")

            if not local_file.exists():
                logger.info(f"Downloading MSMARCO-XI file ({lfile})...")
                url = f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/{lfile}"
                download_success = False
                tmp_file = output_path / f"{Path(lfile).name}.tmp"
                try:
                    import requests
                    headers = {}
                    dl = 0
                    if tmp_file.exists():
                        dl = tmp_file.stat().st_size
                        headers["Range"] = f"bytes={dl}-"
                    
                    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                        if r.status_code in (200, 206):
                            total_sz = int(r.headers.get("content-length", 0)) + (dl if r.status_code == 206 else 0)
                            mode = "ab" if r.status_code == 206 else "wb"
                            if r.status_code == 200:
                                dl = 0
                            with open(tmp_file, mode) as f:
                                for chunk in r.iter_content(chunk_size=1024*1024):
                                    if chunk:
                                        f.write(chunk)
                                        dl += len(chunk)
                                        if total_sz > 0 and (dl % (50*1024*1024) < 1024*1024 or dl == total_sz):
                                            logger.info(f"  Downloaded {dl/(1024*1024):.1f}/{total_sz/(1024*1024):.1f} MB ({(dl*100/total_sz):.1f}%)")
                            tmp_file.rename(local_file)
                            download_success = True
                except Exception as e:
                    logger.warning(f"Direct download failed ({e}), trying hf_hub_download...")
                    try:
                        local_file_path = hf_hub_download(
                            repo_id="ai4bharat/MSMARCO-XI",
                            filename=lfile,
                            repo_type="dataset",
                        )
                        local_file = Path(local_file_path)
                        download_success = True
                    except Exception as e2:
                        logger.warning(f"Failed to download {lfile}: {e2}")

                if not download_success or not local_file.exists():
                    continue

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
    parser.add_argument("--n-passages", type=int, default=10_000)
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    download_and_extract(
        n_passages=args.n_passages,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
