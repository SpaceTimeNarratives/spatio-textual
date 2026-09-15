from spatio_textual.llm_spans import (
    FULL_SPATIAL_LABELS,
    LLMSpanExtractor,
    TOPONYM_LABELS,
    build_span_prompt,
    normalise_model_span,
    span_audit_metrics,
    span_response_schema,
)


class FakeClient:
    provider = "fake"
    model = "fake-model"

    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, task, prompt, *, input_text=None):
        out = dict(self.payload)
        out["telemetry"] = {
            "task": task,
            "backend": "llm",
            "provider": self.provider,
            "model": self.model,
            "latency_ms": 1.0,
            "input_tokens_est": 10,
            "output_tokens_est": 5,
            "cost_usd_est": None,
            "success": True,
            "error": None,
        }
        return out


def test_toponym_schema_is_strict_and_narrow():
    schema = span_response_schema(TOPONYM_LABELS)
    item = schema["properties"]["spans"]["items"]
    assert item["additionalProperties"] is False
    assert item["properties"]["label"]["enum"] == ["TOPONYM"]


def test_full_schema_uses_frozen_gold_label_inventory():
    schema = span_response_schema(FULL_SPATIAL_LABELS)
    assert set(schema["properties"]["spans"]["items"]["properties"]["label"]["enum"]) == set(FULL_SPATIAL_LABELS)


def test_prompt_forbids_offsets_and_modernisation():
    prompt = build_span_prompt("We travelled from Czechoslovakia to London.")
    assert "Do not return character offsets" in prompt
    assert "Preserve historical wording" in prompt
    assert "Return only TOPONYM" in prompt


def test_normalise_span_uses_evidence_to_disambiguate_duplicate_name():
    text = "London was crowded. Later I returned to London by train."
    row = normalise_model_span(
        {
            "text": "London",
            "label": "TOPONYM",
            "evidence_quote": "returned to London by train",
            "certainty": "explicit",
            "confidence": 0.9,
        },
        source_text=text,
        file_id="ex1",
        allowed_labels=TOPONYM_LABELS,
        model="m",
        provider="p",
    )
    assert row["span_grounded"] is True
    assert text[row["start_char"]:row["end_char"]] == "London"
    assert row["start_char"] == text.rfind("London")
    assert row["requires_review"] is False


def test_unique_span_can_be_grounded_even_if_wider_evidence_is_bad():
    text = "We reached Paris before dusk."
    row = normalise_model_span(
        {
            "text": "Paris",
            "label": "TOPONYM",
            "evidence_quote": "We reached Paris at noon.",
            "certainty": "explicit",
            "confidence": 0.8,
        },
        source_text=text,
        file_id="ex2",
        allowed_labels=TOPONYM_LABELS,
        model="m",
        provider="p",
    )
    assert row["span_grounded"] is True
    assert row["evidence_grounded"] is False
    assert row["requires_review"] is True


def test_unmentioned_place_is_not_given_fabricated_offsets():
    text = "We reached the city before dusk."
    row = normalise_model_span(
        {
            "text": "Paris",
            "label": "TOPONYM",
            "evidence_quote": "We reached the city before dusk.",
            "certainty": "contextual_inference",
            "confidence": 0.7,
        },
        source_text=text,
        file_id="ex3",
        allowed_labels=TOPONYM_LABELS,
        model="m",
        provider="p",
    )
    assert row["span_grounded"] is False
    assert row["start_char"] is None
    assert row["end_char"] is None
    assert row["requires_review"] is True


def test_extractor_preserves_raw_response_and_computes_audit():
    client = FakeClient({
        "spans": [
            {
                "text": "Paris",
                "label": "TOPONYM",
                "evidence_quote": "from Paris to London",
                "certainty": "explicit",
                "confidence": 0.95,
            },
            {
                "text": "London",
                "label": "TOPONYM",
                "evidence_quote": "from Paris to London",
                "certainty": "explicit",
                "confidence": 0.95,
            },
        ]
    })
    result = LLMSpanExtractor(client).extract("We moved from Paris to London.", file_id="ex4")
    assert len(result["spans"]) == 2
    assert result["audit"]["unsupported_rate"] == 0.0
    assert result["audit"]["evidence_grounded_rate"] == 1.0
    assert result["prompt_sha256"]
    assert result["raw_structured_response"]["spans"]


def test_span_audit_separates_grounding_and_review():
    audit = span_audit_metrics([
        {"span_grounded": True, "evidence_grounded": True, "requires_review": False, "certainty": "explicit", "label": "TOPONYM"},
        {"span_grounded": False, "evidence_grounded": False, "requires_review": True, "certainty": "ambiguous", "label": None},
    ])
    assert audit["span_grounded_rate"] == 0.5
    assert audit["unsupported_rate"] == 0.5
    assert audit["requires_review_rate"] == 0.5
    assert audit["invalid_label_rate"] == 0.5


def test_failed_span_request_is_not_a_valid_empty_prediction():
    class FailedClient:
        provider = "fake"
        model = "failed-model"

        def complete_json(self, task, prompt, *, input_text=None):
            return {"telemetry": {"success": False, "error": "provider unavailable"}}

    result = LLMSpanExtractor(FailedClient()).extract("We reached Paris.", file_id="failed")
    assert result["spans"] == []
    assert result["backend_error"] is True
    assert result["requires_review"] is True
    assert result["review_reasons"] == ["backend_error"]
    assert result["audit"]["backend_error"] is True
    assert any("provider unavailable" in note for note in result["review_notes"])
