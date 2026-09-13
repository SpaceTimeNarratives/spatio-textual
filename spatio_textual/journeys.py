from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Protocol

from .llm import LLMClient

JOURNEY_FIELDS = ("start_location", "end_location", "transport_mode", "date", "journey_reason")
MODEL_FIELD_STATUSES = {"explicit", "contextual_inference", "missing"}
RUNTIME_FIELD_STATUSES = MODEL_FIELD_STATUSES | {"human_supplied"}


class StructuredJSONClient(Protocol):
    provider: str
    model: str

    def complete_json(self, task: str, prompt: str, *, input_text: str | None = None) -> dict[str, Any]: ...


def build_journey_prompt(source_text: str) -> str:
    """Return the evidence-first journey extraction instruction.

    The prompt deliberately asks for quotations but *not* character offsets.
    Offsets are computed locally from the returned quotation so a model cannot
    fabricate apparently precise provenance.
    """
    schema_example = {
        "journeys": [
            {
                "start_location": "string or null",
                "end_location": "string or null",
                "transport_mode": "string or null",
                "date": "string or null",
                "journey_reason": "string or null",
                "evidence_quote": "exact verbatim substring of Source text",
                "explicit_or_inferred": {
                    "start_location": "explicit | contextual_inference | missing",
                    "end_location": "explicit | contextual_inference | missing",
                    "transport_mode": "explicit | contextual_inference | missing",
                    "date": "explicit | contextual_inference | missing",
                    "journey_reason": "explicit | contextual_inference | missing",
                },
                "confidence": "number 0..1 or null",
                "notes": ["optional short note"],
            }
        ]
    }
    return (
        "Extract journeys or spatial transitions from the Source text.\n"
        "Return one JSON object and no prose outside JSON.\n\n"
        "Rules:\n"
        "1. A journey needs evidence of movement/transition; a place mention alone is not a journey.\n"
        "2. Do not invent missing origin, destination, transport, date or reason. Use null + status 'missing'.\n"
        "3. Use status 'explicit' only when the field is directly expressed in the Source text.\n"
        "4. Use 'contextual_inference' when the field comes from discourse context/anaphora rather than the same explicit movement phrase.\n"
        "5. evidence_quote MUST be copied verbatim from the Source text and should be the smallest passage sufficient to support the journey.\n"
        "6. Do not return character offsets. The software computes offsets from the quotation.\n"
        "7. If no journey is supported, return {\"journeys\": []}.\n"
        "8. Do not convert a historical place name into a modern country name. Preserve the source wording.\n"
        "9. Confidence concerns the extraction, not the historical truth of the narrative.\n\n"
        f"Required JSON shape:\n{json.dumps(schema_example, ensure_ascii=False, indent=2)}\n\n"
        f"Source text:\n{source_text}"
    )


def _evidence_occurrences(source_text: str, quote: str) -> list[int]:
    if not quote:
        return []
    starts: list[int] = []
    pos = 0
    while True:
        idx = source_text.find(quote, pos)
        if idx < 0:
            break
        starts.append(idx)
        pos = idx + 1
    return starts


def _status_for(field: str, value: Any, raw_status: Any, review_notes: list[str]) -> str:
    status = str(raw_status or "").strip()
    if status in MODEL_FIELD_STATUSES:
        if status == "missing" and value not in (None, ""):
            review_notes.append(f"{field}: model marked field missing but also returned a value; value discarded.")
            return "missing"
        if status in {"explicit", "contextual_inference"} and value in (None, ""):
            review_notes.append(f"{field}: model returned status {status!r} without a value; treated as missing.")
            return "missing"
        return status

    # The model was instructed to use a fixed vocabulary. Preserve auditability
    # by recording the correction rather than silently accepting an unknown tag.
    if value in (None, ""):
        review_notes.append(f"{field}: invalid/missing model status {raw_status!r}; treated as missing.")
        return "missing"
    review_notes.append(
        f"{field}: invalid/missing model status {raw_status!r}; value retained as contextual_inference and requires review."
    )
    return "contextual_inference"


def _coerce_confidence(value: Any, review_notes: list[str]) -> float | None:
    if value in (None, ""):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        review_notes.append(f"Invalid confidence {value!r}; stored as null.")
        return None
    if not 0.0 <= score <= 1.0:
        review_notes.append(f"Confidence {score!r} outside 0..1; clipped for storage and requires review.")
        score = min(1.0, max(0.0, score))
    return round(score, 4)


def _journey_id(file_id: str, seg_id: Any, evidence_start: Any, evidence_end: Any, row: dict[str, Any]) -> str:
    key = "|".join(
        str(x or "")
        for x in (
            file_id,
            seg_id,
            evidence_start,
            evidence_end,
            row.get("start_location"),
            row.get("end_location"),
            row.get("transport_mode"),
            row.get("date"),
            row.get("journey_reason"),
        )
    )
    return f"{file_id}-j-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


def normalise_model_journey(
    raw: dict[str, Any],
    *,
    source_text: str,
    file_id: str,
    seg_id: Any,
    model: str | None,
    provider: str | None,
) -> dict[str, Any]:
    """Convert one model proposal into an auditable runtime journey record."""
    review_notes: list[str] = []
    values: dict[str, Any] = {field: raw.get(field) for field in JOURNEY_FIELDS}
    raw_statuses = raw.get("explicit_or_inferred")
    if not isinstance(raw_statuses, dict):
        raw_statuses = {}
        review_notes.append("Model did not return an explicit_or_inferred object.")

    statuses: dict[str, str] = {}
    for field in JOURNEY_FIELDS:
        statuses[field] = _status_for(field, values[field], raw_statuses.get(field), review_notes)
        if statuses[field] == "missing":
            values[field] = None

    quote = raw.get("evidence_quote")
    if not isinstance(quote, str):
        quote = ""
    starts = _evidence_occurrences(source_text, quote)
    evidence_grounded = len(starts) >= 1
    evidence_start: int | None = starts[0] if starts else None
    evidence_end: int | None = evidence_start + len(quote) if evidence_start is not None else None
    if not quote:
        review_notes.append("No evidence quotation returned by the model.")
    elif not starts:
        review_notes.append("Evidence quotation is not an exact substring of the source text.")
    elif len(starts) > 1:
        review_notes.append(
            f"Evidence quotation occurs {len(starts)} times in the source text; first occurrence selected provisionally."
        )

    notes = raw.get("notes")
    if isinstance(notes, str) and notes.strip():
        review_notes.append(f"Model note: {notes.strip()}")
    elif isinstance(notes, list):
        for note in notes:
            if isinstance(note, str) and note.strip():
                review_notes.append(f"Model note: {note.strip()}")

    confidence = _coerce_confidence(raw.get("confidence"), review_notes)
    inferred = any(status == "contextual_inference" for status in statuses.values())
    requires_review = inferred or not evidence_grounded or len(starts) != 1 or bool(review_notes)

    row = {
        "fileId": file_id,
        "segId": seg_id,
        **values,
        "evidence_quote": quote or None,
        "evidence_start_char": evidence_start,
        "evidence_end_char": evidence_end,
        "evidence_grounded": evidence_grounded,
        "evidence_match_count": len(starts),
        "explicit_or_inferred": statuses,
        "confidence": confidence,
        "model": model,
        "provider": provider,
        "requires_review": requires_review,
        "review_notes": review_notes,
        "human_status": "unreviewed",
        "human_edits": [],
    }
    row["journeyId"] = _journey_id(file_id, seg_id, evidence_start, evidence_end, row)
    return row


def validate_runtime_journey(journey: dict[str, Any], source_text: str) -> list[str]:
    """Return audit errors for a runtime journey record."""
    errors: list[str] = []
    jid = journey.get("journeyId", "<missing-journey-id>")
    statuses = journey.get("explicit_or_inferred")
    if not isinstance(statuses, dict):
        return [f"{jid}: explicit_or_inferred must be an object"]

    for field in JOURNEY_FIELDS:
        status = statuses.get(field)
        value = journey.get(field)
        if status not in RUNTIME_FIELD_STATUSES:
            errors.append(f"{jid}: invalid status for {field}: {status!r}")
        elif status == "missing" and value not in (None, ""):
            errors.append(f"{jid}: {field} marked missing but has a value")
        elif status != "missing" and value in (None, ""):
            errors.append(f"{jid}: {field} marked {status} but has no value")

    quote = journey.get("evidence_quote")
    start = journey.get("evidence_start_char")
    end = journey.get("evidence_end_char")
    if journey.get("evidence_grounded"):
        if not isinstance(quote, str) or not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"{jid}: grounded evidence requires quote and integer offsets")
        elif start < 0 or end < start or end > len(source_text) or source_text[start:end] != quote:
            errors.append(f"{jid}: evidence quote/offset mismatch")
    if any(statuses.get(field) == "contextual_inference" for field in JOURNEY_FIELDS):
        if journey.get("requires_review") is not True:
            errors.append(f"{jid}: contextual inference must require review")
    return errors


class JourneyExtractor:
    """Evidence-first LLM journey extractor.

    The model proposes structured fields and a verbatim quotation. The software
    grounds the quotation and computes offsets locally, records per-field
    explicit/inferred status, and automatically routes inference or provenance
    problems to human review.
    """

    def __init__(self, client: StructuredJSONClient | None = None):
        self.client = client or LLMClient()

    def extract(self, source_text: str, *, file_id: str, seg_id: Any = 0) -> dict[str, Any]:
        prompt = build_journey_prompt(source_text)
        data = self.client.complete_json(
            "journey_extraction",
            prompt,
            input_text=source_text,
        )
        telemetry = data.pop("telemetry", None)
        raw_journeys = data.get("journeys", [])
        top_level_notes: list[str] = []
        backend_error = None
        if isinstance(telemetry, dict) and telemetry.get("success") is False:
            detail = str(telemetry.get("error") or "unspecified provider error").strip()
            backend_error = f"backend_error: LLM journey request failed: {detail}"
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
                model=getattr(self.client, "model", None),
                provider=getattr(self.client, "provider", None),
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
            "journeys": journeys,
            "telemetry": telemetry_list,
            "requires_review": bool(top_level_notes) or any(j.get("requires_review") for j in journeys),
            "review_notes": top_level_notes,
            "review_reasons": ["backend_error"] if backend_error else [],
            "backend_error": backend_error is not None,
        }


def journey_field_status_counts(journeys: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Summarise explicit/inferred/missing statuses for workshop/keynote tables."""
    counts = {field: {status: 0 for status in RUNTIME_FIELD_STATUSES} for field in JOURNEY_FIELDS}
    for journey in journeys:
        statuses = journey.get("explicit_or_inferred") or {}
        for field in JOURNEY_FIELDS:
            status = statuses.get(field)
            if status in counts[field]:
                counts[field][status] += 1
    return counts
