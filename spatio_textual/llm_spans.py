from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Protocol, Sequence

from .gold import SPAN_LABELS

TOPONYM_LABELS = ("TOPONYM",)
FULL_SPATIAL_LABELS = tuple(sorted(SPAN_LABELS))
MODEL_CERTAINTY = {"explicit", "contextual_inference", "ambiguous"}

LABEL_DESCRIPTIONS = {
    "TOPONYM": "named geographical place or named spatial feature",
    "GEONOUN": "common-noun geographical or locative expression",
    "SPATIAL_RELATION": "relational expression such as near, beyond or between",
    "DISTANCE": "distance or proximity expression",
    "DIRECTION": "directional expression",
    "TIME": "temporal expression relevant to the spatial narrative",
    "MOVEMENT_CUE": "word or phrase signalling movement or transition",
    "TRANSPORT_CUE": "word or phrase identifying mode of transport",
    "SUBJECTIVE_DESCRIPTOR": "subjective/evaluative description of place",
    "SENSORY_DESCRIPTOR": "sensory description connected to place",
    "DEICTIC_REFERENCE": "context-dependent spatial expression such as here or there",
}


class StructuredJSONClient(Protocol):
    provider: str
    model: str

    def complete_json(self, task: str, prompt: str, *, input_text: str | None = None) -> dict[str, Any]: ...


def span_response_schema(allowed_labels: Sequence[str]) -> dict[str, Any]:
    labels = [str(x) for x in allowed_labels]
    if not labels:
        raise ValueError("allowed_labels must not be empty")
    unknown = sorted(set(labels) - set(SPAN_LABELS))
    if unknown:
        raise ValueError(f"Unsupported spatial labels: {unknown}")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "label": {"type": "string", "enum": labels},
                        "evidence_quote": {"type": "string"},
                        "certainty": {"type": "string", "enum": sorted(MODEL_CERTAINTY)},
                        "confidence": {
                            "anyOf": [
                                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": ["text", "label", "evidence_quote", "certainty", "confidence"],
                },
            }
        },
        "required": ["spans"],
    }


def build_span_prompt(source_text: str, *, allowed_labels: Sequence[str] = TOPONYM_LABELS) -> str:
    labels = tuple(str(x) for x in allowed_labels)
    schema = span_response_schema(labels)
    definitions = "\n".join(f"- {x}: {LABEL_DESCRIPTIONS[x]}" for x in labels)
    scope = (
        "Return only TOPONYM spans for an apples-to-apples named-place NER comparison. "
        "Exclude generic geo-nouns, dates, movement verbs, relations, people and organisations unless the exact expression itself is a named place."
        if labels == TOPONYM_LABELS
        else "Use the broader spatial ontology below and annotate the smallest literal source span supporting each label."
    )
    return (
        "Extract spatially relevant literal spans from the Source text. Return one JSON object and no prose outside JSON.\n\n"
        f"{scope}\n\nAllowed labels:\n{definitions}\n\nRules:\n"
        "1. Every `text` and `evidence_quote` must be copied verbatim from the Source text.\n"
        "2. The evidence quote must contain the returned span and be long enough to disambiguate repeated occurrences where possible.\n"
        "3. Do not return character offsets; software computes them locally.\n"
        "4. Preserve historical wording; do not modernise, geocode or silently normalise place names.\n"
        "5. Do not infer an unmentioned place name. Every span must correspond to literal source characters.\n"
        "6. certainty=explicit for direct wording, contextual_inference when discourse context is needed, ambiguous when genuinely uncertain.\n"
        "7. Confidence concerns annotation confidence, not historical truth.\n"
        "8. If nothing is supported, return {\"spans\": []}.\n"
        "9. Prefer minimal non-overlapping spans unless distinct annotations genuinely overlap.\n\n"
        f"Required JSON schema:\n{json.dumps(schema, ensure_ascii=False)}\n\nSource text:\n{source_text}"
    )


def _occurrences(text: str, phrase: str) -> list[int]:
    if not phrase:
        return []
    out: list[int] = []
    pos = 0
    while True:
        idx = text.find(phrase, pos)
        if idx < 0:
            return out
        out.append(idx)
        pos = idx + 1


def _confidence(value: Any, notes: list[str]) -> float | None:
    if value in (None, ""):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        notes.append(f"Invalid confidence {value!r}; stored as null.")
        return None
    if not 0 <= score <= 1:
        notes.append(f"Confidence {score!r} outside 0..1; clipped.")
        score = min(1.0, max(0.0, score))
    return round(score, 4)


def normalise_model_span(
    raw: dict[str, Any],
    *,
    source_text: str,
    file_id: str,
    allowed_labels: Sequence[str],
    model: str | None,
    provider: str | None,
) -> dict[str, Any]:
    notes: list[str] = []
    allowed = set(allowed_labels)
    text = raw.get("text") if isinstance(raw.get("text"), str) else ""
    label_raw = raw.get("label")
    label = str(label_raw) if label_raw is not None else None
    label_valid = label in allowed
    if not label_valid:
        notes.append(f"Unsupported model label {label_raw!r}.")

    certainty_raw = str(raw.get("certainty") or "")
    certainty = certainty_raw if certainty_raw in MODEL_CERTAINTY else "ambiguous"
    if certainty_raw not in MODEL_CERTAINTY:
        notes.append(f"Invalid/missing certainty {raw.get('certainty')!r}; stored as ambiguous.")

    quote = raw.get("evidence_quote") if isinstance(raw.get("evidence_quote"), str) else ""
    quote_starts = _occurrences(source_text, quote)
    text_starts = _occurrences(source_text, text)
    start: int | None = None

    if quote and len(quote_starts) == 1 and text:
        local = _occurrences(quote, text)
        if len(local) == 1:
            start = quote_starts[0] + local[0]
        elif not local:
            notes.append("Span text is not contained in the evidence quote.")
        else:
            notes.append("Span text occurs multiple times inside the evidence quote.")

    if start is None and len(text_starts) == 1:
        start = text_starts[0]
        if not quote_starts:
            notes.append("Span text grounded uniquely, but evidence quote is not grounded.")

    if not text:
        notes.append("Empty span text cannot be grounded.")
    elif not text_starts:
        notes.append("Span text is not an exact source substring.")
    elif len(text_starts) > 1 and start is None:
        notes.append(f"Span text occurs {len(text_starts)} times and evidence did not disambiguate it.")

    if not quote:
        notes.append("No evidence quote returned.")
    elif not quote_starts:
        notes.append("Evidence quote is not an exact source substring.")
    elif len(quote_starts) > 1:
        notes.append(f"Evidence quote occurs {len(quote_starts)} times.")

    end = start + len(text) if start is not None else None
    grounded = start is not None and bool(text)
    confidence = _confidence(raw.get("confidence"), notes)
    requires_review = not grounded or not label_valid or certainty != "explicit" or len(quote_starts) != 1 or bool(notes)
    key = "|".join(str(x if x is not None else "") for x in (file_id, start, end, text, label))

    return {
        "span_id": f"{file_id}-s-{hashlib.sha1(key.encode()).hexdigest()[:12]}",
        "layer": "llm_spatial_span",
        "label": label if label_valid else None,
        "model_label": label_raw,
        "text": text or None,
        "start_char": start,
        "end_char": end,
        "certainty": certainty,
        "attributes": {},
        "notes": None,
        "evidence_quote": quote or None,
        "evidence_grounded": bool(quote and quote_starts),
        "evidence_match_count": len(quote_starts),
        "span_grounded": grounded,
        "span_match_count": len(text_starts),
        "confidence": confidence,
        "model": model,
        "provider": provider,
        "requires_review": requires_review,
        "review_notes": notes,
        "human_status": "unreviewed",
        "human_edits": [],
    }


def span_audit_metrics(spans: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(spans)
    n = len(rows)
    def count(predicate):
        return sum(1 for row in rows if predicate(row))
    grounded = count(lambda r: r.get("span_grounded") is True)
    return {
        "spans_total": n,
        "span_grounded_rate": round(grounded / n, 6) if n else 1.0,
        "unsupported_rate": round((n - grounded) / n, 6) if n else 0.0,
        "evidence_grounded_rate": round(count(lambda r: r.get("evidence_grounded") is True) / n, 6) if n else 1.0,
        "requires_review_rate": round(count(lambda r: r.get("requires_review") is True) / n, 6) if n else 0.0,
        "contextual_inference_rate": round(count(lambda r: r.get("certainty") == "contextual_inference") / n, 6) if n else 0.0,
        "ambiguous_rate": round(count(lambda r: r.get("certainty") == "ambiguous") / n, 6) if n else 0.0,
        "invalid_label_rate": round(count(lambda r: r.get("label") is None) / n, 6) if n else 0.0,
    }


class LLMSpanExtractor:
    def __init__(self, client: StructuredJSONClient, *, allowed_labels: Sequence[str] = TOPONYM_LABELS):
        self.client = client
        self.allowed_labels = tuple(str(x) for x in allowed_labels)
        span_response_schema(self.allowed_labels)

    def extract(self, source_text: str, *, file_id: str) -> dict[str, Any]:
        prompt = build_span_prompt(source_text, allowed_labels=self.allowed_labels)
        task = "llm_toponym_extraction" if self.allowed_labels == TOPONYM_LABELS else "llm_spatial_span_extraction"
        data = self.client.complete_json(task, prompt, input_text=source_text)
        telemetry = data.pop("telemetry", None)
        raw_text = data.pop("_raw_response_text", None)
        response_meta = data.pop("_response_metadata", None)
        raw_spans = data.get("spans", [])
        review_notes: list[str] = []
        telemetry_rows = [telemetry] if isinstance(telemetry, dict) else (list(telemetry) if isinstance(telemetry, list) else [])
        backend_error = next(
            (
                f"backend_error: LLM span request failed: {str(item.get('error') or 'unspecified provider error').strip()}"
                for item in telemetry_rows
                if isinstance(item, dict) and item.get("success") is False
            ),
            None,
        )
        if backend_error:
            review_notes.append(backend_error)
        if not isinstance(raw_spans, list):
            raw_spans = []
            review_notes.append("Model response `spans` was not a list.")

        spans = [
            normalise_model_span(
                raw,
                source_text=source_text,
                file_id=file_id,
                allowed_labels=self.allowed_labels,
                model=getattr(self.client, "model", None),
                provider=getattr(self.client, "provider", None),
            )
            for raw in raw_spans
            if isinstance(raw, dict)
        ]
        if len(spans) != len(raw_spans):
            review_notes.append("One or more non-object span proposals were ignored.")
        audit = span_audit_metrics(spans)
        audit["backend_error"] = backend_error is not None
        return {
            "schema_version": "spatio-textual-llm-spans-v1",
            "allowed_labels": list(self.allowed_labels),
            "spans": spans,
            "raw_structured_response": data,
            "raw_response_text": raw_text,
            "response_metadata": response_meta,
            "telemetry": telemetry_rows,
            "audit": audit,
            "requires_review": bool(review_notes) or any(x.get("requires_review") for x in spans),
            "review_reasons": ["backend_error"] if backend_error else [],
            "backend_error": backend_error is not None,
            "review_notes": review_notes,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
