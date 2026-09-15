from __future__ import annotations

from typing import Any, Protocol

SENTIMENT_LABELS = ("positive", "negative", "neutral", "mixed")
EMOTION_LABELS = (
    "fear",
    "sadness",
    "anger",
    "joy",
    "anxiety",
    "despair",
    "gratitude",
    "surprise",
)


class StructuredAffectClient(Protocol):
    provider: str
    model: str

    def complete_json(self, task: str, prompt: str, *, input_text: str | None = None) -> dict[str, Any]: ...


def affect_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "sentiment": {"type": "string", "enum": list(SENTIMENT_LABELS)},
            "emotion_labels": {
                "type": "array",
                "items": {"type": "string", "enum": list(EMOTION_LABELS)},
                "uniqueItems": True,
            },
            "evidence_quote": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
            },
            "explicit_or_inferred": {
                "type": "string",
                "enum": ["explicit", "contextual_inference", "none"],
            },
            "confidence": {
                "anyOf": [
                    {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    {"type": "null"},
                ]
            },
            "notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "sentiment",
            "emotion_labels",
            "evidence_quote",
            "explicit_or_inferred",
            "confidence",
            "notes",
        ],
    }


def build_affect_prompt(text: str) -> str:
    labels = ", ".join(EMOTION_LABELS)
    return f"""You are annotating textual affect for a Spatial Humanities research benchmark.

Classify only what the supplied passage supports. A model label is an interpretation of textual evidence, not a direct measurement of a person's hidden psychological state.

Return exactly one JSON object matching the supplied schema.

Sentiment labels:
- positive
- negative
- neutral
- mixed

Emotion labels are multi-label and restricted to:
{labels}

Rules:
1. Keep sentiment and emotion separate. Surprise can be sentiment-neutral, and a passage can be mixed in sentiment.
2. Use an empty emotion_labels array when none of the permitted emotions is sufficiently supported.
3. Do not infer emotion merely from historical or domain terms such as place names, camp, ghetto, war, journey, or archive.
4. Respect negation and questions. A word such as 'fear' mentioned as a title, quotation label, or unanswered question is not automatically evidence that the passage expresses fear.
5. Reported affect counts only when the passage actually attributes the affect to a person or textual source, not when an emotion word is merely mentioned.
6. If affect is explicit, set explicit_or_inferred to 'explicit'. If it depends on contextual behaviour or implication, use 'contextual_inference'. If no emotion is assigned and the passage is affectively neutral, use 'none'.
7. evidence_quote must be a verbatim quotation from the passage that grounds the assigned affect. It may span the whole passage when multiple clauses are needed. Use null only when sentiment is neutral and emotion_labels is empty.
8. Do not invent text, events, motives, or feelings not supported by the passage.
9. confidence is a value from 0 to 1 reflecting confidence in this textual classification, not confidence about historical truth.
10. Keep notes short and use an empty list when no note is needed.

Passage:
{text}
"""


def _ground_quote(text: str, quote: str | None) -> tuple[int | None, int | None, str, list[str]]:
    if quote is None:
        return None, None, "missing", []
    if not isinstance(quote, str) or not quote:
        return None, None, "unsupported", ["Evidence quote is empty or invalid."]
    starts: list[int] = []
    offset = 0
    while True:
        idx = text.find(quote, offset)
        if idx < 0:
            break
        starts.append(idx)
        offset = idx + 1
    if len(starts) == 1:
        start = starts[0]
        return start, start + len(quote), "grounded", []
    if not starts:
        return None, None, "unsupported", ["Evidence quote was not found verbatim in the source text."]
    return None, None, "ambiguous", ["Evidence quote occurs multiple times in the source text."]


def _backend_error(telemetry: Any) -> str | None:
    """Return an auditable error note when structured inference failed."""
    if not isinstance(telemetry, dict) or telemetry.get("success") is not False:
        return None
    detail = str(telemetry.get("error") or "unspecified provider error").strip()
    return f"backend_error: LLM affect request failed: {detail}"


def extract_audited_affect(
    client: StructuredAffectClient,
    text: str,
    *,
    example_id: str,
) -> dict[str, Any]:
    prompt = build_affect_prompt(text)
    data = client.complete_json("affect_classification", prompt, input_text=text)
    telemetry = data.pop("telemetry", None)
    raw_response_text = data.pop("_raw_response_text", None)
    response_metadata = data.pop("_response_metadata", None)

    backend_error = _backend_error(telemetry)
    review_notes: list[str] = []
    invalid_response = False

    raw_sentiment = data.get("sentiment")
    sentiment = raw_sentiment.strip().lower() if isinstance(raw_sentiment, str) else "neutral"
    if not backend_error and sentiment not in SENTIMENT_LABELS:
        review_notes.append(f"Invalid sentiment label {raw_sentiment!r}; treated as neutral.")
        sentiment = "neutral"
        invalid_response = True

    raw_emotions = data.get("emotion_labels")
    emotions: list[str] = []
    if isinstance(raw_emotions, list):
        for value in raw_emotions:
            label = value.strip().lower() if isinstance(value, str) else ""
            if label in EMOTION_LABELS:
                emotions.append(label)
            elif not backend_error:
                review_notes.append(f"Invalid emotion label {value!r}; discarded.")
                invalid_response = True
    elif raw_emotions is not None and not backend_error:
        review_notes.append("emotion_labels must be a list; invalid value discarded.")
        invalid_response = True

    quote = data.get("evidence_quote")
    if quote is not None and not isinstance(quote, str):
        if not backend_error:
            review_notes.append("evidence_quote must be a string or null; invalid value discarded.")
            invalid_response = True
        quote = None

    raw_status = data.get("explicit_or_inferred")
    status = raw_status.strip().lower() if isinstance(raw_status, str) else "none"
    if not backend_error and status not in {"explicit", "contextual_inference", "none"}:
        review_notes.append(f"Invalid explicit_or_inferred value {raw_status!r}; treated as none.")
        status = "none"
        invalid_response = True

    confidence = data.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        if not backend_error:
            review_notes.append(f"Invalid confidence {confidence!r}; stored as null.")
            invalid_response = True
        confidence = None

    raw_notes = data.get("notes")
    notes = [value for value in raw_notes if isinstance(value, str)] if isinstance(raw_notes, list) else []
    if raw_notes is not None and (
        not isinstance(raw_notes, list) or len(notes) != len(raw_notes)
    ) and not backend_error:
        review_notes.append("notes must be a list of strings; invalid values discarded.")
        invalid_response = True

    start, end, grounding_status, grounding_notes = _ground_quote(text, quote)
    review_notes.extend(grounding_notes)
    if backend_error:
        review_notes.append(backend_error)
    if (sentiment != "neutral" or emotions) and quote is None:
        review_notes.append("Non-neutral affect classification has no evidence quote.")
    if not emotions and status == "contextual_inference":
        review_notes.append("Contextual inference status was supplied without an emotion label.")
    if emotions and status == "none":
        review_notes.append("Emotion labels were supplied with explicit_or_inferred='none'.")
    if status == "contextual_inference":
        review_notes.append("Contextual affect inference requires scholarly review.")

    return {
        "example_id": example_id,
        "sentiment_label": sentiment,
        "emotion_labels": sorted(dict.fromkeys(emotions)),
        "evidence_quote": quote,
        "evidence_start_char": start,
        "evidence_end_char": end,
        "evidence_grounding_status": grounding_status,
        "explicit_or_inferred": status,
        "confidence": confidence,
        "notes": notes,
        "requires_review": bool(review_notes),
        "review_reasons": (["backend_error"] if backend_error else []) + (["invalid_response"] if invalid_response else []),
        "backend_error": backend_error is not None,
        "invalid_response": invalid_response,
        "review_notes": review_notes,
        "telemetry": [] if telemetry is None else [telemetry],
        "raw_structured_response": data,
        "raw_response_text": raw_response_text,
        "response_metadata": response_metadata,
        "prompt": prompt,
    }
