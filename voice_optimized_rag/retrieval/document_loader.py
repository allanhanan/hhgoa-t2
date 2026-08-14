"""Document loading and chunking for ingestion into the vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voice_optimized_rag.utils.logging import get_logger
from voice_optimized_rag.retrieval.chunking import chunk_document

logger = get_logger("document_loader")


@dataclass
class DocumentChunk:
    """A chunk of a document ready for embedding."""
    text: str
    metadata: dict


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks using recursive character splitting.

    Tries to split on paragraph boundaries first, then sentences, then words.
    """
    return [
        chunk.text
        for chunk in chunk_document(
            text,
            strategy="sentence",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    ]


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Recursively split text using progressively finer separators."""
    if not text.strip():
        return []

    # Find the best separator that actually exists in the text
    sep = ""
    for s in separators:
        if s in text:
            sep = s
            break

    if not sep:
        # No separator found — hard-split by chunk_size
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    parts = text.split(sep)
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            # If this single part is too large, split it with the next separator
            if len(part) > chunk_size:
                remaining_seps = separators[separators.index(sep) + 1 :] if sep in separators else []
                if remaining_seps:
                    chunks.extend(_recursive_split(part, remaining_seps, chunk_size, chunk_overlap))
                else:
                    for i in range(0, len(part), chunk_size - chunk_overlap):
                        chunk = part[i : i + chunk_size]
                        if chunk.strip():
                            chunks.append(chunk.strip())
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())

    # Add overlap between chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append(prev_tail + " " + chunks[i])
        chunks = overlapped

    return chunks


def load_text_file(path: Path) -> list[DocumentChunk]:
    """Load a text file and split into chunks."""
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    return [
        DocumentChunk(
            text=chunk,
            metadata={"source": str(path), "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]


def load_directory(
    directory: Path,
    extensions: set[str] | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    chunking_strategy: str = "sentence",
    parent_chunk_size: int = 2048,
) -> list[DocumentChunk]:
    """Load all supported files from a directory.

    Args:
        directory: Path to directory.
        extensions: File extensions to include (default: .txt, .md, .rst).
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between adjacent chunks.

    Returns:
        List of DocumentChunk objects.
    """
    if extensions is None:
        extensions = {".txt", ".md", ".rst", ".text"}

    all_chunks: list[DocumentChunk] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            try:
                text = path.read_text(encoding="utf-8")
                chunks = chunk_document(
                    text,
                    strategy=chunking_strategy,  # type: ignore[arg-type]
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    metadata={"source": str(path)},
                    parent_chunk_size=parent_chunk_size,
                )
                for chunk in chunks:
                    all_chunks.append(DocumentChunk(
                        text=chunk.text,
                        metadata=chunk.metadata,
                    ))
                logger.info(f"Loaded {path.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    logger.info(f"Total: {len(all_chunks)} chunks from {directory}")
    return all_chunks
