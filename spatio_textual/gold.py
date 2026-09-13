from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

SPAN_LABELS = {
    "TOPONYM",
    "GEONOUN",
    "SPATIAL_RELATION",
    "DISTANCE",
    "DIRECTION",
    "TIME",
    "MOVEMENT_CUE",
    "TRANSPORT_CUE",
    "SUBJECTIVE_DESCRIPTOR",
    "SENSORY_DESCRIPTOR",
    "DEICTIC_REFERENCE",
}

SPAN_CERTAINTY = {"explicit", "contextual_inference", "ambiguous", "human_supplied"}
JOURNEY_FIELD_STATUS = {"explicit", "contextual_inference", "missing", "human_supplied"}
REFERENCE_STATUS = {
    "draft",
    "adjudicated_reference",
    "provisional_until_source_citation_verified",
    "deprecated",
}
JOURNEY_FIELDS = ("start_location", "end_location", "transport_mode", "date", "journey_reason")
GOLD_SCHEMA_VERSION = "spatio-textual-gold-0.1"


def load_gold_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load one spatio-textual reference record per JSONL line.

    Blank lines and lines beginning with ``#`` are ignored so instructor notes
    can be added outside records during development if needed.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Gold line {line_no} must decode to an object/dict")
            records.append(item)
    return records


def _slice_matches(text: str, start: Any, end: Any, expected: Any) -> bool:
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    if start < 0 or end < start or end > len(text):
        return False
    return text[start:end] == expected


def validate_gold_record(record: dict[str, Any]) -> list[str]:
    """Return human-readable validation errors for one reference record.

    The validator is intentionally dependency-free so that tutorial notebooks,
    CI and the Streamlit demo can all run it in a lightweight environment.
    """
    errors: list[str] = []
    example_id = record.get("example_id", "<missing-example-id>")
    text = record.get("text")

    if record.get("schema_version") != GOLD_SCHEMA_VERSION:
        errors.append(f"{example_id}: unsupported/missing schema_version")
    if not isinstance(example_id, str) or not example_id.strip():
        errors.append("record: example_id must be a non-empty string")
    if not isinstance(text, str):
        errors.append(f"{example_id}: text must be a string")
        text = ""
    if record.get("reference_status") not in REFERENCE_STATUS:
        errors.append(f"{example_id}: invalid reference_status {record.get('reference_status')!r}")

    source = record.get("source")
    if not isinstance(source, dict):
        errors.append(f"{example_id}: source must be an object")
    else:
        if not source.get("distribution_status"):
            errors.append(f"{example_id}: source.distribution_status is required")
        if not source.get("source_note"):
            errors.append(f"{example_id}: source.source_note is required")

    spans = record.get("spans", [])
    if not isinstance(spans, list):
        errors.append(f"{example_id}: spans must be a list")
        spans = []

    span_ids: set[str] = set()
    for idx, span in enumerate(spans):
        prefix = f"{example_id}: spans[{idx}]"
        if not isinstance(span, dict):
            errors.append(f"{prefix} must be an object")
            continue
        sid = span.get("span_id")
        if not isinstance(sid, str) or not sid:
            errors.append(f"{prefix}: span_id is required")
        elif sid in span_ids:
            errors.append(f"{prefix}: duplicate span_id {sid}")
        else:
            span_ids.add(sid)
        if span.get("label") not in SPAN_LABELS:
            errors.append(f"{prefix}: unsupported label {span.get('label')!r}")
        if span.get("certainty") not in SPAN_CERTAINTY:
            errors.append(f"{prefix}: invalid certainty {span.get('certainty')!r}")
        if not _slice_matches(text, span.get("start_char"), span.get("end_char"), span.get("text")):
            errors.append(
                f"{prefix}: text/offset mismatch for {span.get('text')!r} "
                f"at {span.get('start_char')}:{span.get('end_char')}"
            )

    relations = record.get("relations", [])
    if not isinstance(relations, list):
        errors.append(f"{example_id}: relations must be a list")
        relations = []

    relation_ids: set[str] = set()
    for idx, rel in enumerate(relations):
        prefix = f"{example_id}: relations[{idx}]"
        if not isinstance(rel, dict):
            errors.append(f"{prefix} must be an object")
            continue
        rid = rel.get("relation_id")
        if not isinstance(rid, str) or not rid:
            errors.append(f"{prefix}: relation_id is required")
        elif rid in relation_ids:
            errors.append(f"{prefix}: duplicate relation_id {rid}")
        else:
            relation_ids.add(rid)
        if not rel.get("type"):
            errors.append(f"{prefix}: type is required")
        if rel.get("certainty") not in SPAN_CERTAINTY:
            errors.append(f"{prefix}: invalid certainty {rel.get('certainty')!r}")
        for key in ("source_span_id", "target_span_id"):
            sid = rel.get(key)
            if sid is not None and sid not in span_ids:
                errors.append(f"{prefix}: {key}={sid!r} does not resolve to a span")
        if rel.get("source_span_id") is None and not rel.get("source_ref"):
            errors.append(f"{prefix}: provide source_span_id or source_ref")
        if rel.get("target_span_id") is None and not rel.get("target_ref"):
            errors.append(f"{prefix}: provide target_span_id or target_ref")
        if not _slice_matches(
            text,
            rel.get("evidence_start_char"),
            rel.get("evidence_end_char"),
            rel.get("evidence_quote"),
        ):
            errors.append(f"{prefix}: evidence quote/offset mismatch")

    journeys = record.get("journeys", [])
    if not isinstance(journeys, list):
        errors.append(f"{example_id}: journeys must be a list")
        journeys = []

    journey_ids: set[str] = set()
    for idx, journey in enumerate(journeys):
        prefix = f"{example_id}: journeys[{idx}]"
        if not isinstance(journey, dict):
            errors.append(f"{prefix} must be an object")
            continue
        jid = journey.get("journeyId")
        if not isinstance(jid, str) or not jid:
            errors.append(f"{prefix}: journeyId is required")
        elif jid in journey_ids:
            errors.append(f"{prefix}: duplicate journeyId {jid}")
        else:
            journey_ids.add(jid)
        if journey.get("fileId") != example_id:
            errors.append(f"{prefix}: fileId must equal example_id")
        if not _slice_matches(
            text,
            journey.get("evidence_start_char"),
            journey.get("evidence_end_char"),
            journey.get("evidence_quote"),
        ):
            errors.append(f"{prefix}: evidence quote/offset mismatch")

        statuses = journey.get("explicit_or_inferred")
        if not isinstance(statuses, dict):
            errors.append(f"{prefix}: explicit_or_inferred must be an object")
            statuses = {}
        for field in JOURNEY_FIELDS:
            status = statuses.get(field)
            if status not in JOURNEY_FIELD_STATUS:
                errors.append(f"{prefix}: invalid/missing status for {field}: {status!r}")
                continue
            value = journey.get(field)
            if status == "missing" and value not in (None, ""):
                errors.append(f"{prefix}: {field} is marked missing but has value {value!r}")
            if status in {"explicit", "contextual_inference", "human_supplied"} and value in (None, ""):
                errors.append(f"{prefix}: {field} is {status} but has no value")
        if any(statuses.get(field) == "contextual_inference" for field in JOURNEY_FIELDS):
            if journey.get("requires_review") is not True:
                errors.append(f"{prefix}: contextual inference must set requires_review=true")

    return errors


def validate_gold_records(records: Sequence[dict[str, Any]]) -> list[str]:
    """Validate multiple records, including cross-record ID uniqueness."""
    errors: list[str] = []
    ids: set[str] = set()
    for record in records:
        rid = record.get("example_id")
        if isinstance(rid, str) and rid in ids:
            errors.append(f"duplicate example_id across records: {rid}")
        if isinstance(rid, str):
            ids.add(rid)
        errors.extend(validate_gold_record(record))
    return errors


def assert_valid_gold(records: Sequence[dict[str, Any]]) -> None:
    """Raise ``ValueError`` when one or more validation errors are found."""
    errors = validate_gold_records(records)
    if errors:
        preview = "\n".join(f"- {e}" for e in errors)
        raise ValueError(f"Invalid spatio-textual reference data:\n{preview}")


def find_span(
    text: str,
    phrase: str,
    label: str,
    *,
    occurrence: int = 1,
    layer: str | None = None,
    span_id: str | None = None,
    certainty: str = "explicit",
    attributes: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Convenience helper for manual workshop annotation.

    ``occurrence`` is one-based. The helper refuses to guess when the requested
    occurrence is not present, which keeps source offsets auditable.
    """
    if occurrence < 1:
        raise ValueError("occurrence must be >= 1")
    starts: list[int] = []
    pos = 0
    while True:
        idx = text.find(phrase, pos)
        if idx < 0:
            break
        starts.append(idx)
        pos = idx + 1
    if len(starts) < occurrence:
        raise ValueError(f"Could not find occurrence {occurrence} of {phrase!r}")
    start = starts[occurrence - 1]
    if label not in SPAN_LABELS:
        raise ValueError(f"Unsupported spatial span label: {label}")
    return {
        "span_id": span_id,
        "layer": layer,
        "label": label,
        "text": phrase,
        "start_char": start,
        "end_char": start + len(phrase),
        "certainty": certainty,
        "attributes": attributes or {},
        "notes": notes,
    }


def _span_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    a0, a1 = int(a["start_char"]), int(a["end_char"])
    b0, b1 = int(b["start_char"]), int(b["end_char"])
    inter = max(0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union else 0.0


def score_span_annotations(
    predicted: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    *,
    match: str = "exact",
    label_sensitive: bool = True,
) -> dict[str, Any]:
    """Score human/model spans against a reference with one-to-one matching.

    ``match='exact'`` requires identical boundaries. ``match='overlap'`` uses a
    deterministic greedy maximum-IoU matching for overlapping spans. Both modes
    can be label-sensitive (default) or boundary-only.
    """
    if match not in {"exact", "overlap"}:
        raise ValueError("match must be 'exact' or 'overlap'")

    candidates: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(predicted):
        for ri, ref in enumerate(reference):
            if label_sensitive and pred.get("label") != ref.get("label"):
                continue
            if match == "exact":
                ok = (
                    pred.get("start_char") == ref.get("start_char")
                    and pred.get("end_char") == ref.get("end_char")
                )
                score = 1.0 if ok else 0.0
            else:
                try:
                    score = _span_iou(pred, ref)
                except Exception:
                    score = 0.0
            if score > 0:
                candidates.append((score, pi, ri))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_pred: set[int] = set()
    used_ref: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, pi, ri in candidates:
        if pi in used_pred or ri in used_ref:
            continue
        used_pred.add(pi)
        used_ref.add(ri)
        matches.append({"pred_index": pi, "ref_index": ri, "overlap_iou": round(score, 6)})

    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "match": match,
        "label_sensitive": label_sensitive,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "matches": matches,
        "unmatched_pred_indices": [i for i in range(len(predicted)) if i not in used_pred],
        "unmatched_ref_indices": [i for i in range(len(reference)) if i not in used_ref],
    }


def score_relation_annotations(
    predicted: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Exact relation score over relation type and endpoint identifiers/refs."""
    def key(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            item.get("type"),
            item.get("source_span_id"),
            item.get("target_span_id"),
            item.get("source_ref"),
            item.get("target_ref"),
        )

    ref_remaining = list(range(len(reference)))
    matched: list[dict[str, int]] = []
    for pi, pred in enumerate(predicted):
        pkey = key(pred)
        found = None
        for ri in ref_remaining:
            if key(reference[ri]) == pkey:
                found = ri
                break
        if found is not None:
            ref_remaining.remove(found)
            matched.append({"pred_index": pi, "ref_index": found})

    tp = len(matched)
    fp = len(predicted) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "matches": matched,
    }


def select_spans(record: dict[str, Any], labels: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Return reference spans, optionally filtered by label."""
    spans = list(record.get("spans", []))
    if labels is None:
        return spans
    wanted = set(labels)
    return [span for span in spans if span.get("label") in wanted]
