import pytest

from voice_optimized_rag.core.fast_talker import FastTalker


@pytest.mark.asyncio
async def test_fast_talker_refuses_without_context(
    config,
    mock_llm,
    mock_embeddings,
    vector_store,
    cache,
    stream,
    metrics,
) -> None:
    config.fast_talker_fallback_to_retrieval = False
    config.guardrail_min_context_chunks = 1
    fast_talker = FastTalker(
        config=config,
        llm=mock_llm,
        embedding_provider=mock_embeddings,
        vector_store=vector_store,
        cache=cache,
        stream=stream,
        metrics=metrics,
    )

    response = await fast_talker.respond("What is the answer?")

    assert response == config.guardrail_refusal_message
    assert mock_llm.call_count == 0
    assert metrics.get_counter("guardrail_insufficient_context") == 1


@pytest.mark.asyncio
async def test_fast_talker_allows_grounded_context(
    config,
    mock_llm,
    mock_embeddings,
    vector_store,
    cache,
    stream,
    metrics,
) -> None:
    embedding = await mock_embeddings.embed_single("pricing")
    vector_store.add_documents(["Pricing context"], embedding.reshape(1, -1), [{"source": "test"}])
    config.guardrail_min_relevance_score = -1.0
    fast_talker = FastTalker(
        config=config,
        llm=mock_llm,
        embedding_provider=mock_embeddings,
        vector_store=vector_store,
        cache=cache,
        stream=stream,
        metrics=metrics,
    )

    response = await fast_talker.respond("pricing")

    assert response == "Mock response"
    assert mock_llm.call_count == 1
