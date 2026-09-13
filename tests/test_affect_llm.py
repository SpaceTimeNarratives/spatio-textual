from spatio_textual.affect_llm import affect_response_schema, build_affect_prompt, extract_audited_affect


class _FakeClient:
    provider = "openai"
    model = "gpt-test"

    def complete_json(self, task, prompt, *, input_text=None):
        assert task == "affect_classification"
        assert input_text == "Mira felt grateful when the guide returned."
        return {
            "sentiment": "positive",
            "emotion_labels": ["gratitude"],
            "evidence_quote": "felt grateful",
            "explicit_or_inferred": "explicit",
            "confidence": 0.95,
            "notes": [],
            "telemetry": {"success": True, "latency_ms": 10.0},
            "_raw_response_text": '{"sentiment":"positive"}',
            "_response_metadata": {"response_id": "resp_test", "resolved_model": "gpt-test"},
        }


def test_affect_schema_is_strict():
    schema = affect_response_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "sentiment",
        "emotion_labels",
        "evidence_quote",
        "explicit_or_inferred",
        "confidence",
        "notes",
    }
    assert schema["properties"]["emotion_labels"]["uniqueItems"] is True


def test_affect_prompt_separates_textual_evidence_from_psychological_truth():
    prompt = build_affect_prompt("The road crosses the river.")
    assert "not a direct measurement" in prompt
    assert "Keep sentiment and emotion separate" in prompt
    assert "camp" in prompt


def test_audited_affect_grounds_verbatim_evidence():
    text = "Mira felt grateful when the guide returned."
    result = extract_audited_affect(_FakeClient(), text, example_id="a1")
    assert result["sentiment_label"] == "positive"
    assert result["emotion_labels"] == ["gratitude"]
    assert result["evidence_grounding_status"] == "grounded"
    assert text[result["evidence_start_char"]:result["evidence_end_char"]] == "felt grateful"
    assert result["requires_review"] is False
    assert result["response_metadata"]["response_id"] == "resp_test"


def test_contextual_affect_is_routed_to_review():
    class _InferenceClient:
        provider = "openai"
        model = "gpt-test"

        def complete_json(self, task, prompt, *, input_text=None):
            return {
                "sentiment": "negative",
                "emotion_labels": ["anxiety"],
                "evidence_quote": "kept checking the clock",
                "explicit_or_inferred": "contextual_inference",
                "confidence": 0.7,
                "notes": [],
            }

    text = "I kept checking the clock while the announcement was delayed."
    result = extract_audited_affect(_InferenceClient(), text, example_id="a2")
    assert result["evidence_grounding_status"] == "grounded"
    assert result["requires_review"] is True
    assert any("Contextual affect inference" in note for note in result["review_notes"])


def test_failed_affect_request_is_not_a_valid_neutral_prediction():
    class _FailedClient:
        provider = "openai"
        model = "gpt-test"

        def complete_json(self, task, prompt, *, input_text=None):
            return {"telemetry": {"success": False, "error": "provider unavailable"}}

    result = extract_audited_affect(_FailedClient(), "A plain sentence.", example_id="failed")
    assert result["backend_error"] is True
    assert result["requires_review"] is True
    assert result["review_reasons"] == ["backend_error"]
    assert any("provider unavailable" in note for note in result["review_notes"])


def test_malformed_affect_fields_are_rejected_for_review():
    class _MalformedClient:
        provider = "openai"
        model = "gpt-test"

        def complete_json(self, task, prompt, *, input_text=None):
            return {
                "sentiment": "cheerful",
                "emotion_labels": "joy",
                "evidence_quote": 17,
                "explicit_or_inferred": "certain",
                "confidence": "high",
                "notes": "not a list",
                "telemetry": {"success": True},
            }

    result = extract_audited_affect(_MalformedClient(), "A plain sentence.", example_id="bad")
    assert result["sentiment_label"] == "neutral"
    assert result["emotion_labels"] == []
    assert result["evidence_quote"] is None
    assert result["explicit_or_inferred"] == "none"
    assert result["confidence"] is None
    assert result["notes"] == []
    assert result["invalid_response"] is True
    assert result["requires_review"] is True
    assert "invalid_response" in result["review_reasons"]
