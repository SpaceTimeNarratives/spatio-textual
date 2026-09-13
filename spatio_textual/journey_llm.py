from __future__ import annotations

from typing import Any

from .journeys import (
    JOURNEY_FIELDS,
    StructuredJSONClient,
    build_journey_prompt,
    normalise_model_journey,
    validate_runtime_journey,
)


def _nullable_string() -> dict[str, Any]:
    return {"anyOf": [{"type": "string"}, {"type": "null"}]}


def journey_response_schema() -> dict[str, Any]:
    """Strict JSON Schema for evidence-first LLM journey extraction."""
    status_properties = {
        field: {
            "type": "string",
            "enum": ["explicit", "contextual_inference", "missing"],
        }
        for field in JOURNEY_FIELDS
    }
    journey_properties: dict[str, Any] = {
        "start_location": _nullable_string(),
        "end_location": _nullable_string(),
        "transport_mode": _nullable_string(),
        "date": _nullable_string(),
        "journey_reason": _nullable_string(),
        "evidence_quote": {"type": "string"},
        "explicit_or_inferred": {
            "type": "object",
            "additionalProperties": False,
            "properties": status_properties,
            "required": list(JOURNEY_FIELDS),
        },
        "confidence": {
            "anyOf": [
                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                {"type": "null"},
            ]
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "journeys": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": journey_properties,
                    "required": list(journey_properties),
                },
            }
        },
        "required": ["journeys"],
    }


def _backend_error(telemetry: Any) -> str | None:
    """Return an auditable error note when structured inference failed."""
    if not isinstance(telemetry, dict) or telemetry.get("success") is not False:
        return None
    detail = str(telemetry.get("error") or "unspecified provider error").strip()
    return f"backend_error: LLM journey request failed: {detail}"


def extract_audited_journeys(
    client: StructuredJSONClient,
    source_text: str,
    *,
    file_id: str,
    seg_id: Any = 0,
) -> dict[str, Any]:
    """Run strict structured extraction while preserving raw response provenance.

    The model proposes field values plus a verbatim evidence quotation. Software
    grounds that quotation, computes offsets, normalises statuses, and routes
    contextual inference or provenance ambiguity to review.
    """
    prompt = build_journey_prompt(source_text)
    data = client.complete_json("journey_extraction", prompt, input_text=source_text)
    telemetry = data.pop("telemetry", None)
    raw_response_text = data.pop("_raw_response_text", None)
    response_metadata = data.pop("_response_metadata", None)

    raw_journeys = data.get("journeys", [])
    top_level_notes: list[str] = []
    backend_error = _backend_error(telemetry)
    if backend_error:
        top_level_notes.append(backend_error)
    if not isinstance(raw_journeys, list):
        top_level_notes.append("LLM response field 'journeys' was not a list; no journeys accepted.")
        raw_journeys = []

    journeys: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_journeys):
        if not isinstance(raw, dict):
            top_level_notes.append(f"journeys[{index}] was not an object and was skipped.")
            continue
        row = normalise_model_journey(
            raw,
            source_text=source_text,
            file_id=file_id,
            seg_id=seg_id,
            model=getattr(client, "model", None),
            provider=getattr(client, "provider", None),
        )
        errors = validate_runtime_journey(row, source_text)
        if errors:
            row["requires_review"] = True
            row["review_notes"].extend(errors)
        journeys.append(row)

    telemetry_list = [] if telemetry is None else [telemetry]
    return {
        "fileId": file_id,
        "segId": seg_id,
        "text": source_text,
        "prompt": prompt,
        "journeys": journeys,
        "telemetry": telemetry_list,
        "raw_structured_response": data,
        "raw_response_text": raw_response_text,
        "response_metadata": response_metadata,
        "requires_review": bool(top_level_notes) or any(j.get("requires_review") for j in journeys),
        "review_notes": top_level_notes,
        "review_reasons": ["backend_error"] if backend_error else [],
        "backend_error": backend_error is not None,
    }
