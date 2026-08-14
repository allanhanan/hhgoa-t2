from tests.conftest import MockLLM


def test_llm_context_prompt_requires_available_context_answer() -> None:
    messages = MockLLM()._build_messages("What happened?", "Relevant passage")

    assert "Base your answer strictly on the provided context" in messages[0]["content"]
    assert "Do NOT use outside knowledge" in messages[0]["content"]
