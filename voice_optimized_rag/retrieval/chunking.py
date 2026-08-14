"""Configurable document chunking strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


ChunkingStrategy = Literal[
    "sentence",
    "semantic",
    "fixed-token",
    "metadata",
    "parent-child",
]


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict


def chunk_document(
    text: str,
    strategy: ChunkingStrategy = "sentence",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    metadata: dict | None = None,
    parent_chunk_size: int = 2048,
) -> list[Chunk]:
    """Split one document with metadata carried onto every emitted chunk."""
    base = metadata or {}
    if strategy == "fixed-token":
        chunks = _fixed_token_chunks(text, chunk_size, chunk_overlap)
    elif strategy == "semantic":
        chunks = _semantic_chunks(text, chunk_size, chunk_overlap)
    elif strategy == "metadata":
        chunks = _metadata_chunks(text, chunk_size, chunk_overlap)
    elif strategy == "parent-child":
        return _parent_child_chunks(text, chunk_size, chunk_overlap, parent_chunk_size, base)
    else:
        chunks = _sentence_chunks(text, chunk_size, chunk_overlap)

    return [
        Chunk(text=chunk, metadata={**base, "chunk_index": i, "chunk_strategy": strategy})
        for i, chunk in enumerate(chunks)
        if chunk.strip()
    ]


def _sentence_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    return _pack_units([s for s in sentences if s], chunk_size, chunk_overlap)


def _semantic_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
        else:
            units.extend(_sentence_chunks(paragraph, chunk_size, 0))
    return _pack_units(units, chunk_size, chunk_overlap)


def _fixed_token_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    step = max(1, chunk_size - chunk_overlap)
    return [" ".join(tokens[i : i + chunk_size]) for i in range(0, len(tokens), step)]


def _metadata_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sections = re.split(r"(?m)^(#{1,6}\s+.+)$", text)
    units = [s.strip() for s in sections if s.strip()]
    return _pack_units(units, chunk_size, chunk_overlap)


def _parent_child_chunks(
    text: str,
    child_size: int,
    child_overlap: int,
    parent_size: int,
    metadata: dict,
) -> list[Chunk]:
    parents = _sentence_chunks(text, parent_size, 0)
    chunks: list[Chunk] = []
    for parent_idx, parent in enumerate(parents):
        for child_idx, child in enumerate(_sentence_chunks(parent, child_size, child_overlap)):
            chunks.append(Chunk(
                text=child,
                metadata={
                    **metadata,
                    "parent_chunk_index": parent_idx,
                    "chunk_index": child_idx,
                    "chunk_strategy": "parent-child",
                    "parent_text": parent,
                },
            ))
    return chunks


def _pack_units(units: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = unit
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size].strip())
            current = current[max(0, chunk_size - chunk_overlap):]
    if current:
        chunks.append(current)
    if chunk_overlap <= 0 or len(chunks) < 2:
        return chunks
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        overlapped.append(f"{chunks[i - 1][-chunk_overlap:]} {chunks[i]}".strip())
    return overlapped
