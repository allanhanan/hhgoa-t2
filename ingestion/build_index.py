"""Build FAISS binary index + float16 vectors from chunked passages.

Usage:
    python -m ingestion.build_index [--input data/passages.parquet] [--output-dir data/]
    python -m ingestion.build_index --strategy passage_as_chunk
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_index(
    input_path: str = "data/passages.parquet",
    output_dir: str = "data",
    strategy: str = "passage_as_chunk",
    batch_size: int = 512,
    chunk_stream_size: int = 50_000,
):
    """Embed passages, create FAISS binary index, save float16 vectors with low RAM footprint."""
    from app.embedder.encoder import encode_batch, binarize

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    logger.info(f"Building index from {input_path} ({total_rows:,} passages)...")

    dim = 384
    dim_bits = dim

    float16_path = output_path / "float16_vectors.npy"
    # Create memory-mapped numpy array on disk for float16 vectors
    float16_memmap = np.memmap(
        str(float16_path),
        dtype=np.float16,
        mode="w+",
        shape=(total_rows, dim),
    )

    index = faiss.IndexBinaryFlat(dim_bits)
    centroid_sum = np.zeros(dim, dtype=np.float64)

    t0 = time.time()
    processed = 0

    for batch in pf.iter_batches(batch_size=chunk_stream_size, columns=["text"]):
        b_texts = batch.to_pydict()["text"]
        if not b_texts:
            continue

        n_batch = len(b_texts)
        logger.info(f"  Encoding batch {processed:,} .. {processed + n_batch:,} / {total_rows:,}...")

        # Encode float32 [N, 384]
        b_emb = encode_batch(b_texts, batch_size=batch_size)

        # Accumulate centroid sum
        centroid_sum += b_emb.sum(axis=0, dtype=np.float64)

        # Write float16 to memmap
        float16_memmap[processed : processed + n_batch] = b_emb.astype(np.float16)

        # Binarize and add to FAISS
        b_binary = binarize(b_emb)
        index.add(b_binary)

        processed += n_batch

    # Flush memmap to disk
    float16_memmap.flush()
    del float16_memmap

    encode_time = time.time() - t0
    logger.info(f"Encoding complete in {encode_time:.1f}s ({processed / encode_time:.0f} passages/s)")
    logger.info(f"Saved float16 vectors: {float16_path}")

    # Save corpus centroid
    centroid = (centroid_sum / max(processed, 1)).astype(np.float32)
    centroid_path = output_path / "corpus_centroid.npy"
    np.save(str(centroid_path), centroid)
    logger.info(f"Saved corpus centroid: {centroid_path}")

    # Save FAISS binary index
    index_path = output_path / "index.faiss_binary"
    faiss.write_index_binary(index, str(index_path))
    logger.info(f"Saved FAISS index ({index.ntotal:,} vectors): {index_path}")

    return str(index_path), str(float16_path)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS binary index")
    parser.add_argument("--input", type=str, default="data/passages.parquet")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--strategy", type=str, default="passage_as_chunk")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    build_index(
        input_path=args.input,
        output_dir=args.output_dir,
        strategy=args.strategy,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
