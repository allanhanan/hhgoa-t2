"""Build FAISS IndexIVFScalarQuantizer (SQ8) from existing float16 vectors memory-mapped array.

Reduces index memory footprint from 11.4 GB to 2.9 GB for instant loading on machines with <=16GB RAM.

Usage:
    python -m ingestion.build_ivf_index [--float16-path data/float16_vectors.npy] [--output-path data/index.faiss_ivf.bin] [--nlist 8192]
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import faiss
import numpy as np

from app.config import FLOAT16_PATH, IVF_INDEX_PATH, IVF_NLIST, EMBEDDING_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_ivf_index(
    float16_path: str = FLOAT16_PATH,
    output_path: str = IVF_INDEX_PATH,
    nlist: int = IVF_NLIST,
    sample_size_train: int = 200_000,
    add_batch_size: int = 200_000,
    use_sq8: bool = True,
) -> str:
    """Build and train an IVF index (SQ8 or Flat) from memory-mapped float16_vectors.npy."""
    f16_p = Path(float16_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not f16_p.exists():
        raise FileNotFoundError(f"Float16 vector file not found at {f16_p}")

    n_bytes = f16_p.stat().st_size
    dim = EMBEDDING_DIM
    total_rows = n_bytes // (dim * 2)  # float16 = 2 bytes per element
    logger.info(f"Opening memory-mapped float16 vectors at {f16_p} ({total_rows:,} rows, dim={dim}, {n_bytes/(1024**3):.2f} GB)...")

    mmap_vecs = np.memmap(str(f16_p), dtype=np.float16, mode="r", shape=(total_rows, dim))

    # Initialize quantizer and IndexIVFScalarQuantizer using Inner Product metric (cosine similarity for normalized vectors)
    quantizer = faiss.IndexFlatIP(dim)
    if use_sq8:
        logger.info("Using IndexIVFScalarQuantizer (QT_8bit) to reduce RAM footprint to 2.9 GB...")
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, dim, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
        )
    else:
        logger.info("Using IndexIVFFlat...")
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # Step 1: Train IVF index on a bounded sample
    n_train = min(sample_size_train, total_rows)
    step = max(1, total_rows // n_train)
    logger.info(f"Extracting {n_train:,} training sample vectors (subsampling every {step}th row)...")
    
    t0 = time.time()
    train_sample_f16 = mmap_vecs[::step][:n_train]
    train_sample_f32 = train_sample_f16.astype(np.float32)

    logger.info(f"Training FAISS IVF index (nlist={nlist})...")
    index.train(train_sample_f32)
    train_time = time.time() - t0
    logger.info(f"Index training complete in {train_time:.2f}s (is_trained={index.is_trained})")

    # Step 2: Add all vectors in bounded batches to keep RAM usage low
    t_add = time.time()
    added = 0
    for i in range(0, total_rows, add_batch_size):
        end = min(i + add_batch_size, total_rows)
        batch_f16 = mmap_vecs[i:end]
        batch_f32 = batch_f16.astype(np.float32)
        index.add(batch_f32)
        added += (end - i)
        if (i // add_batch_size) % 5 == 0 or end == total_rows:
            logger.info(f"  Added {added:,} / {total_rows:,} vectors to IVF index...")

    add_time = time.time() - t_add
    logger.info(f"Addition complete in {add_time:.2f}s ({added/add_time:.0f} vecs/s). Total index entries: {index.ntotal:,}")

    # Step 3: Save IVF index to disk
    logger.info(f"Saving IVF index to {out_p}...")
    faiss.write_index(index, str(out_p))
    out_size_mb = out_p.stat().st_size / (1024 * 1024)
    logger.info(f"Saved FAISS IVF index successfully: {out_p} ({out_size_mb:.2f} MB)")

    return str(out_p)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS IVF index from float16 vectors")
    parser.add_argument("--float16-path", type=str, default=FLOAT16_PATH)
    parser.add_argument("--output-path", type=str, default=IVF_INDEX_PATH)
    parser.add_argument("--nlist", type=int, default=IVF_NLIST)
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--no-sq8", action="store_true", help="Disable SQ8 quantization")
    args = parser.parse_args()

    build_ivf_index(
        float16_path=args.float16_path,
        output_path=args.output_path,
        nlist=args.nlist,
        sample_size_train=args.sample_size,
        add_batch_size=args.batch_size,
        use_sq8=not args.no_sq8,
    )


if __name__ == "__main__":
    main()
