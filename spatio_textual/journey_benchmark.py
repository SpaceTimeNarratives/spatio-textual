from __future__ import annotations

from typing import Any, Iterable, Sequence

from .benchmark import journey_audit_metrics, summarize_telemetry
from .journey_evaluation import JOURNEY_VALUE_FIELDS, evaluate_journeys


def journey_benchmark_row(
    *,
    example_id: str,
    predicted: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    telemetry: Iterable[dict[str, Any]] = (),
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Evaluate one source passage under the package journey policies.

    Record detection, field accuracy, evidence grounding, contextual inference,
    review burden and telemetry are kept separate. This prevents a fluent but
    weakly grounded structured output from being collapsed into a single score.
    """
    telemetry_rows = [row for row in telemetry if isinstance(row, dict)]
    if any(row.get("success") is False for row in telemetry_rows):
        raise ValueError(f"Cannot score journey backend failure for example_id={example_id!r}")
    evaluation = evaluate_journeys(predicted, reference)
    matching = evaluation["matching"]
    field_scoring = evaluation["field_scoring"]
    audit = journey_audit_metrics(predicted)
    tel = summarize_telemetry(telemetry_rows)
    return {
        "example_id": example_id,
        "provider": provider,
        "model": model,
        "predicted_journeys": len(predicted),
        "reference_journeys": len(reference),
        "tp": matching["tp"],
        "fp": matching["fp"],
        "fn": matching["fn"],
        "precision": matching["precision"],
        "recall": matching["recall"],
        "f1": matching["f1"],
        "matching": matching,
        "field_scoring": field_scoring,
        "evidence_grounded": audit["evidence_grounded"],
        "requires_review": audit["requires_review"],
        "journeys_with_contextual_inference": audit["journeys_with_contextual_inference"],
        "unsupported_rate": audit["unsupported_rate"],
        "evidence_grounded_rate": audit["evidence_grounded_rate"],
        "requires_review_rate": audit["requires_review_rate"],
        "contextual_inference_rate": audit["contextual_inference_rate"],
        "field_contextual_inference_rate": audit["field_contextual_inference_rate"],
        "telemetry_summary": tel,
    }


def _safe_rate(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    return round(numerator / denominator, 6) if denominator else empty


def aggregate_journey_benchmark_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Pool journey metrics over passages without macro-score inflation.

    In particular, passages containing no reference journey and no predicted
    journey do not create artificial perfect precision/recall. Detection metrics
    are computed from pooled TP/FP/FN, while field metrics pool the frozen field
    outcome counts across matched journeys.
    """
    items = list(rows)
    tp = sum(int(row.get("tp") or 0) for row in items)
    fp = sum(int(row.get("fp") or 0) for row in items)
    fn = sum(int(row.get("fn") or 0) for row in items)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    field_counts = {
        field: {
            "correct": 0,
            "missing_prediction": 0,
            "unsupported_prediction": 0,
            "mismatch": 0,
            "both_missing": 0,
        }
        for field in JOURNEY_VALUE_FIELDS
    }
    for row in items:
        fields = ((row.get("field_scoring") or {}).get("fields") or {})
        for field in JOURNEY_VALUE_FIELDS:
            source = fields.get(field) or {}
            for key in field_counts[field]:
                field_counts[field][key] += int(source.get(key) or 0)

    field_summary: dict[str, dict[str, Any]] = {}
    totals = {
        "correct": 0,
        "missing_prediction": 0,
        "unsupported_prediction": 0,
        "mismatch": 0,
        "both_missing": 0,
    }
    for field, counts in field_counts.items():
        for key, value in counts.items():
            totals[key] += value
        predicted_nonmissing = counts["correct"] + counts["unsupported_prediction"] + counts["mismatch"]
        reference_nonmissing = counts["correct"] + counts["missing_prediction"] + counts["mismatch"]
        unsupported_or_wrong = counts["unsupported_prediction"] + counts["mismatch"]
        field_summary[field] = {
            **counts,
            "predicted_nonmissing": predicted_nonmissing,
            "reference_nonmissing": reference_nonmissing,
            "precision": _safe_rate(counts["correct"], predicted_nonmissing, empty=1.0),
            "recall": _safe_rate(counts["correct"], reference_nonmissing, empty=1.0),
            "unsupported_field_rate": _safe_rate(unsupported_or_wrong, predicted_nonmissing),
        }

    predicted_nonmissing = totals["correct"] + totals["unsupported_prediction"] + totals["mismatch"]
    reference_nonmissing = totals["correct"] + totals["missing_prediction"] + totals["mismatch"]
    unsupported_or_wrong = totals["unsupported_prediction"] + totals["mismatch"]
    predicted_total = sum(int(row.get("predicted_journeys") or 0) for row in items)
    reference_total = sum(int(row.get("reference_journeys") or 0) for row in items)
    grounded_total = sum(int(row.get("evidence_grounded") or 0) for row in items)
    review_total = sum(int(row.get("requires_review") or 0) for row in items)
    inferred_total = sum(int(row.get("journeys_with_contextual_inference") or 0) for row in items)

    telemetry_rows = [row.get("telemetry_summary") or {} for row in items]
    calls_total = sum(int(row.get("calls") or 0) for row in telemetry_rows)
    latency_known = sum(int(row.get("latency_ms_known_calls") or 0) for row in telemetry_rows)
    input_known = sum(int(row.get("input_tokens_est_known_calls") or 0) for row in telemetry_rows)
    output_known = sum(int(row.get("output_tokens_est_known_calls") or 0) for row in telemetry_rows)
    latency_total = (
        sum(float(row["latency_ms_total"]) for row in telemetry_rows)
        if calls_total and latency_known == calls_total
        else None
    )
    input_tokens_total = (
        sum(int(row["input_tokens_est_total"]) for row in telemetry_rows)
        if calls_total and input_known == calls_total
        else None
    )
    output_tokens_total = (
        sum(int(row["output_tokens_est_total"]) for row in telemetry_rows)
        if calls_total and output_known == calls_total
        else None
    )
    known_costs = [row.get("cost_usd_est_total") for row in telemetry_rows if row.get("cost_usd_est_total") is not None]

    return {
        "policy": {
            "journey_matching": "spatio-textual-journey-match-v1",
            "field_scoring": "spatio-textual-journey-field-v1",
        },
        "examples": len(items),
        "predicted_journeys": predicted_total,
        "reference_journeys": reference_total,
        "detection": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        },
        "fields": {
            "by_field": field_summary,
            "totals": {
                **totals,
                "predicted_nonmissing": predicted_nonmissing,
                "reference_nonmissing": reference_nonmissing,
                "precision": _safe_rate(totals["correct"], predicted_nonmissing, empty=1.0),
                "recall": _safe_rate(totals["correct"], reference_nonmissing, empty=1.0),
                "unsupported_field_rate": _safe_rate(unsupported_or_wrong, predicted_nonmissing),
            },
        },
        "audit": {
            "evidence_grounded": grounded_total,
            "evidence_grounded_rate": _safe_rate(grounded_total, predicted_total, empty=1.0),
            "requires_review": review_total,
            "requires_review_rate": _safe_rate(review_total, predicted_total),
            "journeys_with_contextual_inference": inferred_total,
            "contextual_inference_rate": _safe_rate(inferred_total, predicted_total),
        },
        "telemetry": {
            "calls": calls_total,
            "latency_ms_total": round(latency_total, 3) if latency_total is not None else None,
            "latency_ms_mean_per_call": round(latency_total / calls_total, 3) if latency_total is not None else None,
            "latency_ms_known_calls": latency_known,
            "input_tokens_est_total": input_tokens_total,
            "input_tokens_est_known_calls": input_known,
            "output_tokens_est_total": output_tokens_total,
            "output_tokens_est_known_calls": output_known,
            "cost_usd_est_total": round(sum(float(x) for x in known_costs), 8) if known_costs else None,
            "cost_note": "Cost remains null unless the provider adapter supplies an estimate; token counts are estimates.",
        },
    }
