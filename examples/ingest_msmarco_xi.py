#!/usr/bin/env python3
"""Stream ai4bharat/MSMARCO-XI into Qdrant without loading it into RAM."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_optimized_rag import VORConfig
from voice_optimized_rag.retrieval.embeddings import create_embedding_provider
from voice_optimized_rag.retrieval.qdrant_store import QdrantVectorStore
from voice_optimized_rag.retrieval.chunking import chunk_document


def _checkpoint_path(path: Path, config_name: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{config_name}.json"


def _read_checkpoint(path: Path) -> int:
    if not path.exists():
        return 0
    return json.loads(path.read_text()).get("rows_seen", 0)


def _write_checkpoint(path: Path, rows_seen: int) -> None:
    path.write_text(json.dumps({"rows_seen": rows_seen}, indent=2))


def _translated_passages(row: dict) -> list[str]:
    passages = row.get("passages") or {}
    translated = (
        passages.get("Translated_passages")
        or row.get("Translated_passages")
        or row.get("translated_passages")
        or []
    )
    if isinstance(translated, dict):
        translated = list(translated.values())
    return [str(p).strip() for p in translated if str(p).strip()]


def _english_passages(row: dict) -> list[str]:
    passages = row.get("passages") or {}
    english = (
        passages.get("English_passages")
        or row.get("English_passages")
        or row.get("english_passages")
        or []
    )
    if isinstance(english, dict):
        english = list(english.values())
    return [str(p).strip() for p in english if str(p).strip()]


def _selected_flags(row: dict) -> list[int]:
    passages = row.get("passages") or {}
    selected = passages.get("is_selected") or row.get("is_selected") or []
    return [int(v) for v in selected]


def _passage_text(row: dict) -> str:
    return "\n".join(_translated_passages(row))


def _iter_passages(
    row: dict,
    selected_only: bool,
    max_chars: int,
    passage_field: str = "auto",
) -> list[tuple[int, str, int | None]]:
    if passage_field == "english":
        texts = _english_passages(row)
    elif passage_field == "translated":
        texts = _translated_passages(row)
    else:
        texts = _translated_passages(row) or _english_passages(row)
    flags = _selected_flags(row)
    passages: list[tuple[int, str, int | None]] = []
    for idx, text in enumerate(texts):
        selected = flags[idx] if idx < len(flags) else None
        if selected_only and selected != 1:
            continue
        if max_chars > 0:
            text = text[:max_chars]
        if text:
            passages.append((idx, text, selected))
    return passages


def load_dev_sample(path: Path) -> list[dict]:
    """Load the checked-in MSMARCO-XI development sample as UTF-8 JSON."""
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return rows


def build_sample_passages(
    rows: list[dict],
    selected_only: bool = False,
    max_chars: int = 0,
    passage_field: str = "auto",
) -> tuple[list[str], list[str], list[dict]]:
    texts: list[str] = []
    ids: list[str] = []
    metadata: list[dict] = []
    seen: set[str] = set()

    for row_index, row in enumerate(rows, 1):
        query_id = str(row.get("query_id", row.get("Query_id", row_index)))
        passages = _iter_passages(row, selected_only, max_chars, passage_field)
        for passage_idx, text, selected in passages:
            base_metadata = {
                "query_id": query_id,
                "query": row.get("query", ""),
                "Eng_Query": row.get("Eng_Query", ""),
                "Answer": row.get("Answer", ""),
                "passage_index": passage_idx,
                "is_selected": selected,
                "source": "data/dev/msmarco_xi_tamil_sample.json",
                "dataset": "ai4bharat/MSMARCO-XI",
                "dataset_config": "tamil",
                "passage_language": "en" if passage_field == "english" else "ta",
                "query_language": "ta",
                "row_index": row_index,
            }
            
            chunks = chunk_document(
                text=text,
                strategy="parent-child",
                chunk_size=300,
                chunk_overlap=50,
                metadata=base_metadata,
                parent_chunk_size=1024,
            )
            
            for chunk in chunks:
                dedupe_key = chunk.text.strip()
                if not dedupe_key or dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                point_key = f"dev-sample:{query_id}:{passage_idx}:{chunk.metadata['chunk_index']}:{dedupe_key}"
                texts.append(chunk.text)
                ids.append(str(uuid5(NAMESPACE_URL, point_key)))
                metadata.append(chunk.metadata)
    return texts, ids, metadata


async def ingest_sample(args: argparse.Namespace) -> int:
    config = VORConfig(
        vector_store_provider="qdrant",
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key or "",
        qdrant_collection=args.collection,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
    )
    embeddings = create_embedding_provider(config)
    if embeddings.dimension != args.embedding_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: expected {args.embedding_dimension}, got {embeddings.dimension}"
        )
    store = QdrantVectorStore(
        dimension=embeddings.dimension,
        url=config.qdrant_url,
        api_key=config.qdrant_api_key or None,
        collection_name=config.qdrant_collection,
    )

    rows = load_dev_sample(args.sample_json)
    texts, ids, metadata = build_sample_passages(
        rows,
        selected_only=args.selected_only,
        max_chars=args.max_chars,
        passage_field=args.passage_field,
    )
    if not texts:
        raise ValueError(f"No passages found in {args.sample_json}")
    embedded = await embeddings.embed(texts)
    store.add_documents(texts, embedded, metadata, ids=ids, batch_size=args.upsert_batch_size)
    print(f"Upserted {len(texts)} dev-sample passages into {args.collection}")
    return len(texts)


async def ingest_config(args: argparse.Namespace, config_name: str) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    config = VORConfig(
        vector_store_provider="qdrant",
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key or "",
        qdrant_collection=args.collection,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
    )
    embeddings = create_embedding_provider(config)
    store = QdrantVectorStore(
        dimension=embeddings.dimension,
        url=config.qdrant_url,
        api_key=config.qdrant_api_key or None,
        collection_name=config.qdrant_collection,
    )

    ckpt = _checkpoint_path(args.checkpoint_dir, config_name)
    skip_rows = _read_checkpoint(ckpt)
    dataset = load_dataset(args.dataset, config_name, split=args.split, streaming=True)

    texts: list[str] = []
    ids: list[str] = []
    metadata: list[dict] = []
    rows_seen = 0
    upserted = 0
    stop_after_flush = False

    for row in dataset:
        if args.langs:
            row_lang = row.get("target_lang", config_name)
            if row_lang not in args.langs:
                continue

        rows_seen += 1
        if rows_seen <= skip_rows:
            continue
        passages = _iter_passages(row, args.selected_only, args.max_chars)
        if not passages:
            _write_checkpoint(ckpt, rows_seen)
            continue

        query_id = str(row.get("query_id", row.get("Query_id", rows_seen)))
        for passage_idx, text, selected in passages:
            if args.max_passages and upserted + len(texts) >= args.max_passages:
                stop_after_flush = True
                break
                
            base_metadata = {
                "query_id": query_id,
                "passage_index": passage_idx,
                "is_selected": selected,
                "query_type": row.get("query_type", row.get("Query_type")),
                "target_lang": row.get("target_lang", config_name),
                "dataset_config": config_name,
                "row_index": rows_seen,
            }
            
            chunks = chunk_document(
                text=text,
                strategy="parent-child",
                chunk_size=300,
                chunk_overlap=50,
                metadata=base_metadata,
                parent_chunk_size=1024,
            )
            
            for chunk in chunks:
                point_key = f"{config_name}:{query_id}:{rows_seen}:{passage_idx}:{chunk.metadata['chunk_index']}"
                texts.append(chunk.text)
                ids.append(str(uuid5(NAMESPACE_URL, point_key)))
                metadata.append(chunk.metadata)

        if texts and (len(texts) >= args.batch_size or stop_after_flush):
            batch_embeddings = await embeddings.embed(texts)
            store.add_documents(texts, batch_embeddings, metadata, ids=ids, batch_size=args.upsert_batch_size)
            upserted += len(texts)
            texts, ids, metadata = [], [], []
            _write_checkpoint(ckpt, rows_seen)
            print(f"{config_name}: rows_seen={rows_seen} upserted={upserted}", flush=True)

        if stop_after_flush:
            break

        if args.max_rows and rows_seen >= args.max_rows:
            break

    if texts:
        batch_embeddings = await embeddings.embed(texts)
        store.add_documents(texts, batch_embeddings, metadata, ids=ids, batch_size=args.upsert_batch_size)
        upserted += len(texts)
    _write_checkpoint(ckpt, rows_seen)
    return upserted


async def main() -> None:
    parser = argparse.ArgumentParser(description="Stream MSMARCO-XI passages into Qdrant")
    parser.add_argument("--sample-json", type=Path, help="Ingest a local MSMARCO-XI JSON sample instead of streaming HF")
    parser.add_argument("--dataset", default="ai4bharat/MSMARCO-XI")
    parser.add_argument("--configs", nargs="+", help="HF dataset config names/languages")
    parser.add_argument("--langs", nargs="+", help="Filter by target_lang (e.g., en hi ta)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--upsert-batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-passages", type=int, default=0, help="Stop after this many passages; 0 disables")
    parser.add_argument("--max-chars", type=int, default=600, help="Trim each passage before embedding; 0 disables")
    parser.add_argument("--selected-only", action="store_true", help="Only ingest passages marked is_selected=1")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/msmarco_xi_checkpoints"))
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-api-key", default="")
    parser.add_argument("--collection", default="msmarco_xi")
    parser.add_argument("--embedding-provider", default="sentence-transformers")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--embedding-dimension", type=int, default=384)
    parser.add_argument("--passage-field", choices=["translated", "english", "auto"], default="auto")
    args = parser.parse_args()

    if args.sample_json:
        await ingest_sample(args)
        return
    if not args.configs:
        parser.error("--configs is required unless --sample-json is provided")

    total = 0
    for config_name in args.configs:
        total += await ingest_config(args, config_name)
    print(f"Upserted {total} passages")


if __name__ == "__main__":
    asyncio.run(main())
