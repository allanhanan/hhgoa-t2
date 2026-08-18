"""Build SQLite payload database from chunked passages.

Usage:
    python -m ingestion.build_payload_db [--input data/passages.parquet] [--output data/payloads.db]
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_payload_db(
    input_path: str = "data/passages.parquet",
    output_path: str = "data/payloads.db",
):
    """Create SQLite database mapping vector IDs to passage text + metadata."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing DB
    if output.exists():
        output.unlink()

    # Create DB
    conn = sqlite3.connect(str(output))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")

    conn.execute("""
        CREATE TABLE payloads (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            query_type TEXT DEFAULT '',
            source_query TEXT DEFAULT '',
            is_selected INTEGER DEFAULT 0
        )
    """)

    logger.info(f"Streaming passages from {input_path} into SQLite payloads table...")
    pf = pq.ParquetFile(input_path)
    total_inserted = 0

    for batch in pf.iter_batches(batch_size=50_000):
        b_dict = batch.to_pydict()
        ids = b_dict["id"]
        texts = b_dict["text"]
        qtypes = b_dict.get("query_type", [""] * len(ids))
        squeries = b_dict.get("source_query", [""] * len(ids))
        selecteds = b_dict.get("is_selected", [0] * len(ids))

        rows = [
            (ids[i], texts[i], qtypes[i], squeries[i], selecteds[i])
            for i in range(len(ids))
        ]
        conn.executemany(
            "INSERT INTO payloads (id, text, query_type, source_query, is_selected) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        total_inserted += len(rows)
        if total_inserted % 500_000 == 0:
            logger.info(f"  Inserted {total_inserted} rows...")

    # Verify
    count = conn.execute("SELECT COUNT(*) FROM payloads").fetchone()[0]
    logger.info(f"Payload DB created: {count} rows at {output}")

    conn.close()
    return str(output)


def main():
    parser = argparse.ArgumentParser(description="Build SQLite payload database")
    parser.add_argument("--input", type=str, default="data/passages.parquet")
    parser.add_argument("--output", type=str, default="data/payloads.db")
    args = parser.parse_args()

    build_payload_db(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
