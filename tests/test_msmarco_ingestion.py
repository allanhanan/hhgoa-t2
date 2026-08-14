from pathlib import Path

from examples.ingest_msmarco_xi import (
    _iter_passages,
    _passage_text,
    build_sample_passages,
    load_dev_sample,
)


def test_msmarco_ingestion_uses_translated_passages() -> None:
    row = {
        "passages": {"Translated_passages": ["canonical passage"]},
        "Answer": "do not use this as corpus",
    }

    assert _passage_text(row) == "canonical passage"


def test_msmarco_ingestion_can_filter_selected_passages() -> None:
    row = {
        "passages": {
            "Translated_passages": ["selected", "not selected"],
            "is_selected": [1, 0],
        }
    }

    assert _iter_passages(row, selected_only=True, max_chars=0) == [(0, "selected", 1)]


def test_dev_sample_loads_utf8() -> None:
    rows = load_dev_sample(Path("data/dev/msmarco_xi_tamil_sample.json"))

    assert len(rows) == 10
    assert rows[0]["query"] == "மன்ஹாட்டன் திட்டத்தின் வெற்றியின் உடனடி விளைவு என்ன?"
    assert rows[0]["passages"]["English_passages"][0].startswith("The presence of communication")


def test_dev_sample_builds_deduped_english_payloads() -> None:
    rows = load_dev_sample(Path("data/dev/msmarco_xi_tamil_sample.json"))

    texts, ids, metadata = build_sample_passages(rows, passage_field="english")

    assert texts
    assert len(texts) == len(set(texts))
    assert len(ids) == len(texts)
    assert metadata[0]["query_id"] == "1185869"
    assert metadata[0]["query"]
    assert metadata[0]["Eng_Query"]
    assert metadata[0]["Answer"]
    assert metadata[0]["passage_language"] == "en"
