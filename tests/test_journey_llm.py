from spatio_textual.journey_llm import extract_audited_journeys, journey_response_schema


class _FakeClient:
    provider = "openai"
    model = "gpt-test"

    def complete_json(self, task, prompt, *, input_text=None):
        assert task == "journey_extraction"
        assert input_text == "Mira travelled from Aberdeen to Inverness by train."
        return {
            "journeys": [
                {
                    "start_location": "Aberdeen",
                    "end_location": "Inverness",
                    "transport_mode": "train",
                    "date": None,
                    "journey_reason": None,
                    "evidence_quote": "Mira travelled from Aberdeen to Inverness by train.",
                    "explicit_or_inferred": {
                        "start_location": "explicit",
                        "end_location": "explicit",
                        "transport_mode": "explicit",
                        "date": "missing",
                        "journey_reason": "missing",
                    },
                    "confidence": 0.98,
                    "notes": [],
                }
            ],
            "telemetry": {"success": True, "input_tokens": 10, "output_tokens": 20},
            "_raw_response_text": '{"journeys":[]}',
            "_response_metadata": {"response_id": "resp_test", "resolved_model": "gpt-test"},
        }


def test_journey_schema_is_strict_and_requires_all_runtime_fields():
    schema = journey_response_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["journeys"]
    item = schema["properties"]["journeys"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {
        "start_location",
        "end_location",
        "transport_mode",
        "date",
        "journey_reason",
        "evidence_quote",
        "explicit_or_inferred",
        "confidence",
        "notes",
    }
    statuses = item["properties"]["explicit_or_inferred"]
    assert statuses["additionalProperties"] is False
    assert set(statuses["required"]) == {
        "start_location",
        "end_location",
        "transport_mode",
        "date",
        "journey_reason",
    }


def test_audited_journey_extraction_preserves_raw_and_grounds_evidence():
    text = "Mira travelled from Aberdeen to Inverness by train."
    result = extract_audited_journeys(_FakeClient(), text, file_id="x1")
    assert len(result["journeys"]) == 1
    journey = result["journeys"][0]
    assert journey["start_location"] == "Aberdeen"
    assert journey["end_location"] == "Inverness"
    assert journey["transport_mode"] == "train"
    assert journey["evidence_grounded"] is True
    assert text[journey["evidence_start_char"]:journey["evidence_end_char"]] == journey["evidence_quote"]
    assert result["raw_response_text"] == '{"journeys":[]}'
    assert result["response_metadata"]["response_id"] == "resp_test"
    assert result["telemetry"][0]["success"] is True


def test_failed_journey_request_is_not_a_valid_empty_response():
    class _FailedClient:
        provider = "openai"
        model = "gpt-test"

        def complete_json(self, task, prompt, *, input_text=None):
            return {"telemetry": {"success": False, "error": "provider unavailable"}}

    result = extract_audited_journeys(_FailedClient(), "We went home.", file_id="failed")
    assert result["journeys"] == []
    assert result["backend_error"] is True
    assert result["requires_review"] is True
    assert result["review_reasons"] == ["backend_error"]
    assert any("provider unavailable" in note for note in result["review_notes"])
