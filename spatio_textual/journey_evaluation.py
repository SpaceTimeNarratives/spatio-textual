from __future__ import annotations

import re
from typing import Any, Sequence

JOURNEY_VALUE_FIELDS = (
    "start_location",
    "end_location",
    "transport_mode",
    "date",
    "journey_reason",
)

TRANSPORT_ALIASES = {
    "walking": "foot",
    "walk": "foot",
    "walked": "foot",
    "on foot": "foot",
    "by foot": "foot",
    "foot": "foot",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "cycling": "bicycle",
    "cycled": "bicycle",
    "plane": "air",
    "airplane": "air",
    "flight": "air",
    "flying": "air",
    "flew": "air",
    "air": "air",
    "train": "train",
    "rail": "train",
    "bus": "bus",
    "coach": "bus",
    "car": "car",
    "lorry": "lorry",
    "truck": "lorry",
    "boat": "boat",
    "ship": "ship",
    "ferry": "ferry",
}


def normalize_journey_value(value: Any, field: str | None = None) -> str | None:
    """Apply transparent lexical normalization for journey evaluation only."""
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip()).casefold()
    if not text:
        return None
    if field == "transport_mode":
        # Remove a transparent leading preposition before alias lookup.
        candidate = re.sub(r"^(?:by|on)\s+", "", text)
        return TRANSPORT_ALIASES.get(text, TRANSPORT_ALIASES.get(candidate, candidate))
    return text


def journey_values_equal(reference: Any, predicted: Any, field: str) -> bool:
    left = normalize_journey_value(reference, field)
    right = normalize_journey_value(predicted, field)
    return left is not None and left == right


def evidence_iou(reference: dict[str, Any], predicted: dict[str, Any]) -> float:
    """Return character-offset IoU for two evidence spans, or 0 when unavailable."""
    a0, a1 = reference.get("evidence_start_char"), reference.get("evidence_end_char")
    b0, b1 = predicted.get("evidence_start_char"), predicted.get("evidence_end_char")
    if not all(isinstance(value, int) for value in (a0, a1, b0, b1)):
        return 0.0
    if a1 <= a0 or b1 <= b0:
        return 0.0
    inter = max(0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union else 0.0


def _endpoint_match_counts(reference: dict[str, Any], predicted: dict[str, Any]) -> tuple[int, int]:
    expected = 0
    matched = 0
    for field in ("start_location", "end_location"):
        if normalize_journey_value(reference.get(field), field) is None:
            continue
        expected += 1
        if journey_values_equal(reference.get(field), predicted.get(field), field):
            matched += 1
    return matched, expected


def candidate_journey_match(
    reference: dict[str, Any],
    predicted: dict[str, Any],
    *,
    evidence_threshold: float = 0.50,
) -> dict[str, Any]:
    """Describe whether one prediction is eligible to match one reference journey."""
    endpoint_matches, endpoint_expected = _endpoint_match_counts(reference, predicted)
    overlap = evidence_iou(reference, predicted)
    all_expected_endpoints_match = endpoint_expected > 0 and endpoint_matches == endpoint_expected
    evidence_supported_partial = overlap >= evidence_threshold and endpoint_matches >= 1
    evidence_only_match = endpoint_expected == 0 and overlap >= evidence_threshold
    eligible = all_expected_endpoints_match or evidence_supported_partial or evidence_only_match

    exact_quote = False
    ref_quote = reference.get("evidence_quote")
    pred_quote = predicted.get("evidence_quote")
    if isinstance(ref_quote, str) and isinstance(pred_quote, str):
        exact_quote = ref_quote == pred_quote

    # Endpoint agreement dominates. Evidence overlap disambiguates repeated
    # movements with the same places; exact quotation is only a tie-breaker.
    score = (3.0 * endpoint_matches) + (2.0 * overlap) + (0.1 if exact_quote else 0.0)
    return {
        "eligible": eligible,
        "score": round(score, 6),
        "endpoint_matches": endpoint_matches,
        "endpoint_expected": endpoint_expected,
        "evidence_iou": round(overlap, 6),
        "exact_evidence_quote": exact_quote,
        "eligibility_reason": (
            "all_reference_endpoints_match"
            if all_expected_endpoints_match
            else "evidence_overlap_plus_endpoint"
            if evidence_supported_partial
            else "evidence_overlap_no_endpoints"
            if evidence_only_match
            else "not_eligible"
        ),
    }


def match_journeys(
    predicted: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    *,
    evidence_threshold: float = 0.50,
) -> dict[str, Any]:
    """Greedily select deterministic one-to-one journey matches."""
    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    for pi, pred in enumerate(predicted):
        for ri, ref in enumerate(reference):
            detail = candidate_journey_match(ref, pred, evidence_threshold=evidence_threshold)
            if detail["eligible"]:
                candidates.append((float(detail["score"]), pi, ri, detail))

    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    used_pred: set[int] = set()
    used_ref: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, pi, ri, detail in candidates:
        if pi in used_pred or ri in used_ref:
            continue
        used_pred.add(pi)
        used_ref.add(ri)
        matches.append({
            "pred_index": pi,
            "ref_index": ri,
            **detail,
        })

    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "policy": "spatio-textual-journey-match-v1",
        "evidence_threshold": evidence_threshold,
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


def _empty_field_counts() -> dict[str, int]:
    return {
        "correct": 0,
        "missing_prediction": 0,
        "unsupported_prediction": 0,
        "mismatch": 0,
        "both_missing": 0,
    }


def score_matched_journey_fields(
    predicted: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    matches: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Score structured fields for already matched journey records."""
    by_field = {field: _empty_field_counts() for field in JOURNEY_VALUE_FIELDS}

    for match in matches:
        pred = predicted[int(match["pred_index"])]
        ref = reference[int(match["ref_index"])]
        for field in JOURNEY_VALUE_FIELDS:
            ref_value = normalize_journey_value(ref.get(field), field)
            pred_value = normalize_journey_value(pred.get(field), field)
            counts = by_field[field]
            if ref_value is None and pred_value is None:
                counts["both_missing"] += 1
            elif ref_value is None and pred_value is not None:
                counts["unsupported_prediction"] += 1
            elif ref_value is not None and pred_value is None:
                counts["missing_prediction"] += 1
            elif ref_value == pred_value:
                counts["correct"] += 1
            else:
                counts["mismatch"] += 1

    totals = _empty_field_counts()
    field_metrics: dict[str, dict[str, Any]] = {}
    for field, counts in by_field.items():
        for key, value in counts.items():
            totals[key] += value
        predicted_nonmissing = counts["correct"] + counts["unsupported_prediction"] + counts["mismatch"]
        reference_nonmissing = counts["correct"] + counts["missing_prediction"] + counts["mismatch"]
        unsupported_or_wrong = counts["unsupported_prediction"] + counts["mismatch"]
        field_metrics[field] = {
            **counts,
            "predicted_nonmissing": predicted_nonmissing,
            "reference_nonmissing": reference_nonmissing,
            "precision": round(counts["correct"] / predicted_nonmissing, 6) if predicted_nonmissing else 1.0,
            "recall": round(counts["correct"] / reference_nonmissing, 6) if reference_nonmissing else 1.0,
            "unsupported_field_rate": round(unsupported_or_wrong / predicted_nonmissing, 6) if predicted_nonmissing else 0.0,
        }

    predicted_nonmissing = totals["correct"] + totals["unsupported_prediction"] + totals["mismatch"]
    reference_nonmissing = totals["correct"] + totals["missing_prediction"] + totals["mismatch"]
    unsupported_or_wrong = totals["unsupported_prediction"] + totals["mismatch"]
    return {
        "policy": "spatio-textual-journey-field-v1",
        "matched_journeys": len(matches),
        "fields": field_metrics,
        "totals": {
            **totals,
            "predicted_nonmissing": predicted_nonmissing,
            "reference_nonmissing": reference_nonmissing,
            "precision": round(totals["correct"] / predicted_nonmissing, 6) if predicted_nonmissing else 1.0,
            "recall": round(totals["correct"] / reference_nonmissing, 6) if reference_nonmissing else 1.0,
            "unsupported_field_rate": round(unsupported_or_wrong / predicted_nonmissing, 6) if predicted_nonmissing else 0.0,
        },
    }


def evaluate_journeys(
    predicted: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    *,
    evidence_threshold: float = 0.50,
) -> dict[str, Any]:
    """Run record matching followed by exact-normalized field evaluation."""
    matching = match_journeys(predicted, reference, evidence_threshold=evidence_threshold)
    fields = score_matched_journey_fields(predicted, reference, matching["matches"])
    return {
        "matching": matching,
        "field_scoring": fields,
    }
