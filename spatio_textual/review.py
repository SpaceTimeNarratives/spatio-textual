from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable

HUMAN_STATUSES = {"unreviewed", "accepted", "edited", "rejected"}
REVIEW_REASONS = {
    "model_disagreement",
    "ambiguous_place",
    "unresolved_place",
    "low_confidence",
    "contextual_inference",
    "unsupported_llm_field",
    "segmentation_exception",
    "backend_error",
    "human_flag",
    "disambiguation",
    "correction",
}


def _timestamp(value: str | None = None) -> str:
    return value or datetime.now(timezone.utc).isoformat()


def apply_human_review(
    record: dict[str, Any],
    *,
    action: str,
    field: str | None = None,
    new_value: Any = None,
    reason: str = "human_flag",
    editor: str = "session_user",
    timestamp: str | None = None,
    resolve_review: bool = True,
) -> dict[str, Any]:
    """Return a reviewed copy while preserving an append-only human edit trail.

    Supported actions are ``accept``, ``edit`` and ``reject``. For ``edit``, a
    field name is required. The original machine value is retained in the edit
    event before the runtime record is updated.
    """
    action = str(action).strip().lower()
    if action not in {"accept", "edit", "reject"}:
        raise ValueError("action must be one of: accept, edit, reject")
    if reason not in REVIEW_REASONS:
        raise ValueError(f"Unsupported review reason: {reason}")
    if action == "edit" and not field:
        raise ValueError("field is required when action='edit'")

    out = deepcopy(record)
    edits = list(out.get("human_edits") or [])
    event: dict[str, Any] = {
        "timestamp": _timestamp(timestamp),
        "field": field,
        "old_value": out.get(field) if field else None,
        "new_value": new_value if action == "edit" else None,
        "action": action,
        "reason": reason,
        "editor": editor,
    }
    edits.append(event)
    out["human_edits"] = edits

    if action == "edit":
        out[field] = new_value
        out["human_status"] = "edited"
    elif action == "accept":
        out["human_status"] = "accepted"
    else:
        out["human_status"] = "rejected"

    if resolve_review:
        out["requires_review"] = False

    return out


def apply_place_review(record: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Review a place, invalidating geocoding after a resolved-name edit.

    A corrected name alone does not establish coordinates. Preserve the previous
    resolution in the append-only edit trail and keep the place reviewable.
    """
    out = apply_human_review(record, **kwargs)
    action = str(kwargs.get("action") or "").strip().lower()
    if action != "edit" or kwargs.get("field") != "resolved_name":
        return out
    cleared = {
        "lat": None, "lon": None, "latitude": None, "longitude": None,
        "place_type_resolved": None, "geo_source": None, "geo_confidence": None,
        "geonameid": None, "countrycode": None,
        "candidates": [], "candidates_count": 0, "ambiguous": False,
        "resolution_status": "unresolved",
        "review_reason": "Place name edited; coordinates require verification.",
    }
    for field, value in cleared.items():
        if field in out or field in {"lat", "lon", "resolution_status", "review_reason"}:
            out = apply_human_review(
                out, action="edit", field=field, new_value=value,
                reason="correction", editor=kwargs.get("editor", "session_user"),
                timestamp=kwargs.get("timestamp"), resolve_review=False,
            )
    out["requires_review"] = True
    return out


def human_correction_burden(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarise human review/correction burden for benchmark tables.

    ``correction_rate`` counts edited or rejected records. Acceptance is tracked
    separately because confirming a suggestion still consumes review effort but
    does not imply that the machine output required correction.
    """
    rows = list(records)
    total = len(rows)
    statuses = {status: 0 for status in HUMAN_STATUSES}
    edit_events = 0
    edited_fields: set[str] = set()

    for row in rows:
        status = row.get("human_status", "unreviewed")
        if status not in statuses:
            status = "unreviewed"
        statuses[status] += 1
        for event in row.get("human_edits") or []:
            edit_events += 1
            field = event.get("field")
            if field:
                edited_fields.add(str(field))

    corrected = statuses["edited"] + statuses["rejected"]
    reviewed = total - statuses["unreviewed"]
    return {
        "records_total": total,
        "records_reviewed": reviewed,
        "records_corrected": corrected,
        "review_rate": round(reviewed / total, 4) if total else 0.0,
        "correction_rate": round(corrected / total, 4) if total else 0.0,
        "human_edit_events": edit_events,
        "unique_fields_edited": len(edited_fields),
        "edited_fields": sorted(edited_fields),
        "status_counts": statuses,
    }
