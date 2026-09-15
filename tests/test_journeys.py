from spatio_textual.journeys import (
    JourneyExtractor,
    build_journey_prompt,
    journey_field_status_counts,
    normalise_model_journey,
    validate_runtime_journey,
)


SOURCE = (
    "Q: Where did you live before the war?\n"
    "A: We lived in Amsterdam near my mother's family. Later we were deported by train to Auschwitz."
)


class FakeStructuredClient:
    provider = "fake"
    model = "fake-structured-model"

    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, task, prompt, *, input_text=None):
        data = dict(self.payload)
        data["telemetry"] = {
            "task": task,
            "backend": "llm",
            "provider": self.provider,
            "model": self.model,
            "success": True,
            "latency_ms": 1.0,
            "input_chars": len(input_text or ""),
            "input_tokens_est": 1,
            "output_tokens_est": 1,
            "cost_usd_est": None,
            "error": None,
        }
        return data


def model_journey(evidence_quote=None):
    return {
        "start_location": "Amsterdam",
        "end_location": "Auschwitz",
        "transport_mode": "train",
        "date": None,
        "journey_reason": "deportation",
        "evidence_quote": evidence_quote or "We lived in Amsterdam near my mother's family. Later we were deported by train to Auschwitz.",
        "explicit_or_inferred": {
            "start_location": "contextual_inference",
            "end_location": "explicit",
            "transport_mode": "explicit",
            "date": "missing",
            "journey_reason": "explicit",
        },
        "confidence": 0.91,
        "notes": [],
    }


def test_prompt_requires_verbatim_evidence_and_forbids_model_offsets():
    prompt = build_journey_prompt(SOURCE)
    assert "evidence_quote MUST be copied verbatim" in prompt
    assert "Do not return character offsets" in prompt
    assert "Do not invent missing origin" in prompt
    assert SOURCE in prompt


def test_normaliser_computes_offsets_from_quote_locally():
    row = normalise_model_journey(
        model_journey(),
        source_text=SOURCE,
        file_id="synthetic",
        seg_id=1,
        model="m",
        provider="p",
    )
    assert row["evidence_grounded"] is True
    assert SOURCE[row["evidence_start_char"]:row["evidence_end_char"]] == row["evidence_quote"]
    assert row["explicit_or_inferred"]["start_location"] == "contextual_inference"
    assert row["requires_review"] is True
    assert validate_runtime_journey(row, SOURCE) == []


def test_unsupported_quote_is_never_given_fake_offsets():
    row = normalise_model_journey(
        model_journey("This sentence does not occur in the source."),
        source_text=SOURCE,
        file_id="synthetic",
        seg_id=1,
        model="m",
        provider="p",
    )
    assert row["evidence_grounded"] is False
    assert row["evidence_start_char"] is None
    assert row["evidence_end_char"] is None
    assert row["requires_review"] is True
    assert any("not an exact substring" in note for note in row["review_notes"])


def test_invalid_model_status_is_corrected_openly_and_reviewed():
    raw = model_journey()
    raw["explicit_or_inferred"]["start_location"] = "probably"
    row = normalise_model_journey(
        raw,
        source_text=SOURCE,
        file_id="synthetic",
        seg_id=1,
        model="m",
        provider="p",
    )
    assert row["explicit_or_inferred"]["start_location"] == "contextual_inference"
    assert row["requires_review"] is True
    assert any("invalid/missing model status" in note for note in row["review_notes"])


def test_missing_status_discards_inconsistent_model_value():
    raw = model_journey()
    raw["date"] = "1944"
    raw["explicit_or_inferred"]["date"] = "missing"
    row = normalise_model_journey(
        raw,
        source_text=SOURCE,
        file_id="synthetic",
        seg_id=1,
        model="m",
        provider="p",
    )
    assert row["date"] is None
    assert row["explicit_or_inferred"]["date"] == "missing"
    assert row["requires_review"] is True


def test_extractor_with_fake_client_returns_auditable_record_and_telemetry():
    client = FakeStructuredClient({"journeys": [model_journey()]})
    out = JourneyExtractor(client).extract(SOURCE, file_id="synthetic", seg_id=7)
    assert len(out["journeys"]) == 1
    journey = out["journeys"][0]
    assert journey["fileId"] == "synthetic"
    assert journey["segId"] == 7
    assert journey["model"] == client.model
    assert journey["provider"] == client.provider
    assert journey["journeyId"].startswith("synthetic-j-")
    assert out["telemetry"][0]["task"] == "journey_extraction"
    assert out["requires_review"] is True


def test_no_journey_response_is_valid_and_not_forced_into_schema():
    client = FakeStructuredClient({"journeys": []})
    out = JourneyExtractor(client).extract("There was a river near the town.", file_id="none")
    assert out["journeys"] == []
    assert out["review_notes"] == []
    assert out["requires_review"] is False


def test_extractor_propagates_failed_request_as_backend_error():
    class FailedClient:
        provider = "fake"
        model = "failed-model"

        def complete_json(self, task, prompt, *, input_text=None):
            return {"telemetry": {"success": False, "error": "timeout"}}

    out = JourneyExtractor(FailedClient()).extract(SOURCE, file_id="failed")
    assert out["journeys"] == []
    assert out["backend_error"] is True
    assert out["review_reasons"] == ["backend_error"]
    assert out["requires_review"] is True
    assert any("timeout" in note for note in out["review_notes"])


def test_non_list_journey_payload_is_rejected_at_top_level():
    client = FakeStructuredClient({"journeys": {"start_location": "London"}})
    out = JourneyExtractor(client).extract(SOURCE, file_id="bad")
    assert out["journeys"] == []
    assert out["requires_review"] is True
    assert any("was not a list" in note for note in out["review_notes"])


def test_journey_ids_are_deterministic_for_same_grounded_proposal():
    kwargs = dict(source_text=SOURCE, file_id="synthetic", seg_id=1, model="m", provider="p")
    a = normalise_model_journey(model_journey(), **kwargs)
    b = normalise_model_journey(model_journey(), **kwargs)
    assert a["journeyId"] == b["journeyId"]


def test_field_status_summary_keeps_explicit_inferred_and_missing_separate():
    row = normalise_model_journey(
        model_journey(),
        source_text=SOURCE,
        file_id="synthetic",
        seg_id=1,
        model="m",
        provider="p",
    )
    counts = journey_field_status_counts([row])
    assert counts["start_location"]["contextual_inference"] == 1
    assert counts["end_location"]["explicit"] == 1
    assert counts["date"]["missing"] == 1
