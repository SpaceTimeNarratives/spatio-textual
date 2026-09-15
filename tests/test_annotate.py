from __future__ import annotations

from spatio_textual.annotate import annotate_text


def test_annotate_text_current_schema():
    """The public helper should return the v0.4 audit-ready record shape.

    This test intentionally avoids asserting exact spaCy model predictions: the
    lightweight CI environment may use the blank-pipeline fallback when a spaCy
    language model is not installed. Exact model outputs belong in benchmark
    fixtures, not package smoke tests.
    """
    text = "I travelled from Amsterdam to London."
    result = annotate_text(text, include_entities=True, include_verbs=True)

    assert isinstance(result, dict)
    assert result["text"] == text
    assert isinstance(result.get("entities"), list)
    assert isinstance(result.get("verb_data"), list)
    assert isinstance(result.get("event_data"), list)
    assert isinstance(result.get("telemetry"), list)
    assert isinstance(result.get("review_notes"), list)
    assert "requires_review" in result


def test_entity_records_are_offset_grounded_when_present():
    text = "The journey to Dachau continued."
    result = annotate_text(text, include_entities=True)

    for entity in result.get("entities", []):
        assert isinstance(entity.get("start_char"), int)
        assert isinstance(entity.get("end_char"), int)
        assert entity["start_char"] < entity["end_char"]
        assert text[entity["start_char"] : entity["end_char"]] == entity["text"]
        assert "label" in entity
        assert "source" in entity


def test_annotation_telemetry_is_auditable():
    result = annotate_text("We moved to Amsterdam.")
    telemetry = result.get("telemetry") or []

    assert telemetry
    first = telemetry[0]
    for key in (
        "task",
        "backend",
        "provider",
        "model",
        "latency_ms",
        "input_chars",
        "input_tokens_est",
        "success",
    ):
        assert key in first
