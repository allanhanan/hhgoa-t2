from voice_optimized_rag.retrieval.chunking import chunk_document


def test_sentence_chunking_preserves_metadata() -> None:
    chunks = chunk_document(
        "One sentence. Two sentence. Three sentence.",
        strategy="sentence",
        chunk_size=20,
        metadata={"source": "unit"},
    )

    assert chunks
    assert chunks[0].metadata["source"] == "unit"
    assert chunks[0].metadata["chunk_strategy"] == "sentence"


def test_parent_child_chunking_adds_parent_metadata() -> None:
    chunks = chunk_document(
        "Alpha sentence. Beta sentence. Gamma sentence.",
        strategy="parent-child",
        chunk_size=18,
        parent_chunk_size=40,
    )

    assert chunks
    assert "parent_chunk_index" in chunks[0].metadata
    assert "parent_text" in chunks[0].metadata
