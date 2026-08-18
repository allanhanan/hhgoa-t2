"""Multiple chunking strategies for MSMARCO-XI passages.

Demonstrates 4 approaches as required by the spec:
1. PassageAsChunk — use passages directly (optimal for pre-chunked data)
2. SlidingWindowChunk — fixed-size with overlap
3. SemanticSentenceChunk — split on sentence boundaries with semantic grouping
4. HierarchicalChunk — two-level: passage-level + passage-group

Usage:
    python -m ingestion.chunking_strategies --strategy all --input data/passages.parquet
"""
from __future__ import annotations

import argparse
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    """A single chunk ready for embedding."""
    id: int
    text: str
    metadata: dict = field(default_factory=dict)


class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""

    @abstractmethod
    def chunk(self, passages: list[dict]) -> list[ChunkRecord]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class PassageAsChunk(ChunkingStrategy):
    """Use passages directly — optimal for pre-chunked MSMARCO data.

    MSMARCO passages are already ~56 words, which is near the sweet spot
    for dense retrieval. No further splitting needed.
    """

    @property
    def name(self) -> str:
        return "passage_as_chunk"

    def chunk(self, passages: list[dict]) -> list[ChunkRecord]:
        return [
            ChunkRecord(
                id=p["id"],
                text=p["text"],
                metadata={
                    "strategy": self.name,
                    "query_type": p.get("query_type", ""),
                    "is_selected": p.get("is_selected", 0),
                },
            )
            for p in passages
            if p.get("text", "").strip()
        ]


class SlidingWindowChunk(ChunkingStrategy):
    """Fixed-size sliding window with configurable overlap.

    For passages shorter than window_size, passes them through unchanged.
    For longer passages, creates overlapping windows to ensure no context is lost.
    """

    def __init__(self, window_size: int = 128, overlap: int = 32):
        self.window_size = window_size  # in words
        self.overlap = overlap

    @property
    def name(self) -> str:
        return f"sliding_window_{self.window_size}_{self.overlap}"

    def chunk(self, passages: list[dict]) -> list[ChunkRecord]:
        chunks = []
        chunk_id = 0
        for p in passages:
            text = p.get("text", "").strip()
            if not text:
                continue

            words = text.split()
            if len(words) <= self.window_size:
                chunks.append(ChunkRecord(
                    id=chunk_id,
                    text=text,
                    metadata={
                        "strategy": self.name,
                        "source_id": p["id"],
                        "window_idx": 0,
                    },
                ))
                chunk_id += 1
            else:
                step = self.window_size - self.overlap
                for i in range(0, len(words), step):
                    window = words[i:i + self.window_size]
                    if len(window) < self.overlap:
                        break
                    chunks.append(ChunkRecord(
                        id=chunk_id,
                        text=" ".join(window),
                        metadata={
                            "strategy": self.name,
                            "source_id": p["id"],
                            "window_idx": i // step,
                        },
                    ))
                    chunk_id += 1
        return chunks


class SemanticSentenceChunk(ChunkingStrategy):
    """Split on sentence boundaries, group sentences up to max_sentences.

    Preserves semantic coherence by never splitting mid-sentence.
    """

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences
        self._sent_split = re.compile(r'(?<=[.!?])\s+')

    @property
    def name(self) -> str:
        return f"semantic_sentence_{self.max_sentences}"

    def chunk(self, passages: list[dict]) -> list[ChunkRecord]:
        chunks = []
        chunk_id = 0
        for p in passages:
            text = p.get("text", "").strip()
            if not text:
                continue

            sentences = self._sent_split.split(text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= self.max_sentences:
                chunks.append(ChunkRecord(
                    id=chunk_id,
                    text=text,
                    metadata={
                        "strategy": self.name,
                        "source_id": p["id"],
                        "n_sentences": len(sentences),
                    },
                ))
                chunk_id += 1
            else:
                for i in range(0, len(sentences), self.max_sentences):
                    group = sentences[i:i + self.max_sentences]
                    chunks.append(ChunkRecord(
                        id=chunk_id,
                        text=" ".join(group),
                        metadata={
                            "strategy": self.name,
                            "source_id": p["id"],
                            "sentence_range": f"{i}-{i+len(group)}",
                        },
                    ))
                    chunk_id += 1
        return chunks


class HierarchicalChunk(ChunkingStrategy):
    """Two-level chunking: individual passages + grouped passages by query.

    Creates both fine-grained (passage-level) and coarse (query-group-level)
    chunks to support both precise and broad retrieval.
    """

    def __init__(self, group_size: int = 3, max_group_words: int = 300):
        self.group_size = group_size
        self.max_group_words = max_group_words

    @property
    def name(self) -> str:
        return f"hierarchical_{self.group_size}"

    def chunk(self, passages: list[dict]) -> list[ChunkRecord]:
        chunks = []
        chunk_id = 0

        # Level 1: Individual passages
        for p in passages:
            text = p.get("text", "").strip()
            if not text:
                continue
            chunks.append(ChunkRecord(
                id=chunk_id,
                text=text,
                metadata={
                    "strategy": self.name,
                    "level": "passage",
                    "source_id": p["id"],
                    "is_selected": p.get("is_selected", 0),
                },
            ))
            chunk_id += 1

        # Level 2: Group consecutive passages
        group_texts = []
        for p in passages:
            text = p.get("text", "").strip()
            if not text:
                continue
            group_texts.append(text)
            if len(group_texts) >= self.group_size:
                combined = " ".join(group_texts)
                words = combined.split()
                if len(words) <= self.max_group_words:
                    chunks.append(ChunkRecord(
                        id=chunk_id,
                        text=combined,
                        metadata={
                            "strategy": self.name,
                            "level": "group",
                            "group_size": len(group_texts),
                        },
                    ))
                    chunk_id += 1
                group_texts = []

        return chunks


# Registry of all strategies
STRATEGIES: dict[str, type[ChunkingStrategy]] = {
    "passage_as_chunk": PassageAsChunk,
    "sliding_window": SlidingWindowChunk,
    "semantic_sentence": SemanticSentenceChunk,
    "hierarchical": HierarchicalChunk,
}


def get_strategy(name: str, **kwargs) -> ChunkingStrategy:
    """Get a chunking strategy by name."""
    cls = STRATEGIES.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return cls(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()) + ["all"], default="passage_as_chunk")
    parser.add_argument("--input", type=str, default="data/passages.parquet")
    args = parser.parse_args()

    # Load passages
    table = pq.read_table(args.input)
    passages = table.to_pylist()
    logger.info(f"Loaded {len(passages)} passages")

    strategies = list(STRATEGIES.keys()) if args.strategy == "all" else [args.strategy]

    for sname in strategies:
        strategy = get_strategy(sname)
        chunks = strategy.chunk(passages)
        logger.info(f"  {strategy.name}: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
