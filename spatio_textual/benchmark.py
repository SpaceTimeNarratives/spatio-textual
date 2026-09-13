from __future__ import annotations

from statistics import mean
from typing import Any, Iterable, Sequence

from .gold import score_span_annotations
from .review import human_correction_burden


def summarize_telemetry(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate latency/token/cost telemetry without inventing missing values."""
    items = [row for row in rows if isinstance(row, dict)]
    latency = [float(row["latency_ms"]) for row in items if isinstance(row.get("latency_ms"), (int, float))]
    input_tokens = [int(row["input_tokens_est"]) for row in items if isinstance(row.get("input_tokens_est"), (int, float))]
    output_tokens = [int(row["output_tokens_est"]) for row in items if isinstance(row.get("output_tokens_est"), (int, float))]
    costs = [float(row["cost_usd_est"]) for row in items if isinstance(row.get("cost_usd_est"), (int, float))]
    successes = [bool(row.get("success")) for row in items if row.get("success") is not None]
    return {
        "calls": len(items),
        "latency_ms_total": round(sum(latency), 3) if latency else None,
        "latency_ms_mean": round(mean(latency), 3) if latency else None,
        "latency_ms_known_calls": len(latency),
        "input_tokens_est_total": sum(input_tokens) if input_tokens else None,
        "input_tokens_est_known_calls": len(input_tokens),
        "output_tokens_est_total": sum(output_tokens) if output_tokens else None,
        "output_tokens_est_known_calls": len(output_tokens),
        "cost_usd_est_total": round(sum(costs), 8) if costs else None,
        "success_rate": round(sum(successes) / len(successes), 6) if successes else None,
    }


def span_comparison_row(
    *,
    example_id: str,
    method: str,
    backend: str,
    model: str | None,
    predicted: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    task: str = "spatial_annotation",
    match: str = "exact",
    supported_reference_total: int | None = None,
    reference_total: int | None = None,
    telemetry: Iterable[dict[str, Any]] = (),
    notes: str | None = None,
) -> dict[str, Any]:
    """Build one tidy comparison row for a span-based task.

    ``coverage`` is ontology/representational coverage when both totals are
    supplied. It is deliberately distinct from empirical recall.
    """
    score = score_span_annotations(predicted, reference, match=match, label_sensitive=True)
    tel = summarize_telemetry(telemetry)
    coverage = None
    if reference_total is not None and supported_reference_total is not None:
        coverage = round(supported_reference_total / reference_total, 6) if reference_total else 1.0
    return {
        "example_id": example_id,
        "method": method,
        "backend": backend,
        "model": model,
        "task": task,
        "precision": score["precision"],
        "recall": score["recall"],
        "f1": score["f1"],
        "coverage": coverage,
        "unsupported_rate": None,
        "ambiguous_rate": None,
        "human_edits_required": None,
        "latency_ms": tel["latency_ms_total"],
        "cost_usd_est": tel["cost_usd_est_total"],
        "notes": notes,
        "tp": score["tp"],
        "fp": score["fp"],
        "fn": score["fn"],
        "match": match,
        "telemetry_summary": tel,
    }


def journey_audit_metrics(journeys: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarise evidence/inference/review characteristics of journey output."""
    rows = list(journeys)
    total = len(rows)
    grounded = sum(1 for row in rows if row.get("evidence_grounded") is True)
    review = sum(1 for row in rows if row.get("requires_review") is True)
    inferred = 0
    fields_nonmissing = 0
    fields_inferred = 0
    ambiguous = 0
    for row in rows:
        statuses = row.get("explicit_or_inferred") or {}
        row_inferred = False
        for status in statuses.values():
            if status != "missing":
                fields_nonmissing += 1
            if status == "contextual_inference":
                fields_inferred += 1
                row_inferred = True
        inferred += int(row_inferred)
        if row.get("ambiguous") or row.get("ambiguous_endpoint"):
            ambiguous += 1
    return {
        "journeys_total": total,
        "evidence_grounded": grounded,
        "evidence_grounded_rate": round(grounded / total, 6) if total else 1.0,
        "unsupported_rate": round((total - grounded) / total, 6) if total else 0.0,
        "requires_review": review,
        "requires_review_rate": round(review / total, 6) if total else 0.0,
        "journeys_with_contextual_inference": inferred,
        "contextual_inference_rate": round(inferred / total, 6) if total else 0.0,
        "field_contextual_inference_rate": round(fields_inferred / fields_nonmissing, 6) if fields_nonmissing else 0.0,
        "ambiguous_rate": round(ambiguous / total, 6) if total else 0.0,
    }


def journey_comparison_row(
    *,
    example_id: str,
    method: str,
    backend: str,
    model: str | None,
    journeys: Sequence[dict[str, Any]],
    telemetry: Iterable[dict[str, Any]] = (),
    reviewed_records: Sequence[dict[str, Any]] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Build a tidy audit-oriented comparison row for structured journeys.

    Journey precision/recall remain null here until a task-specific journey
    matching policy is frozen. Evidence grounding and review burden are reported
    immediately because their semantics are already defined by the common schema.
    """
    audit = journey_audit_metrics(journeys)
    tel = summarize_telemetry(telemetry)
    burden = human_correction_burden(reviewed_records or []) if reviewed_records is not None else None
    return {
        "example_id": example_id,
        "method": method,
        "backend": backend,
        "model": model,
        "task": "journey_extraction",
        "precision": None,
        "recall": None,
        "f1": None,
        "coverage": None,
        "unsupported_rate": audit["unsupported_rate"],
        "ambiguous_rate": audit["ambiguous_rate"],
        "human_edits_required": burden["records_corrected"] if burden is not None else None,
        "latency_ms": tel["latency_ms_total"],
        "cost_usd_est": tel["cost_usd_est_total"],
        "notes": notes,
        "evidence_grounded_rate": audit["evidence_grounded_rate"],
        "requires_review_rate": audit["requires_review_rate"],
        "contextual_inference_rate": audit["contextual_inference_rate"],
        "field_contextual_inference_rate": audit["field_contextual_inference_rate"],
        "review_burden": burden,
        "telemetry_summary": tel,
    }


def _micro_scores(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return pooled span scores when rows expose integer TP/FP/FN counts."""
    scored = [
        row for row in group
        if all(isinstance(row.get(field), int) for field in ("tp", "fp", "fn"))
    ]
    if not scored:
        return {
            "tp_total": None,
            "fp_total": None,
            "fn_total": None,
            "precision_micro": None,
            "recall_micro": None,
            "f1_micro": None,
        }
    tp = sum(int(row["tp"]) for row in scored)
    fp = sum(int(row["fp"]) for row in scored)
    fn = sum(int(row["fn"]) for row in scored)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp_total": tp,
        "fp_total": fp,
        "fn_total": fn,
        "precision_micro": round(precision, 6),
        "recall_micro": round(recall, 6),
        "f1_micro": round(f1, 6),
    }


def aggregate_comparison_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate comparison rows with explicit macro/micro and telemetry semantics.

    Backwards-compatible ``precision``/``recall``/``f1`` remain macro means over
    examples. For span tasks, pooled ``*_micro`` metrics are also returned from
    summed TP/FP/FN. Latency and cost expose both mean-per-example and total
    values so keynote figures cannot accidentally describe a mean as a total.
    Null values remain null rather than being zero-filled.
    """
    items = list(rows)
    groups: dict[tuple[str, str, str, str | None], list[dict[str, Any]]] = {}
    for row in items:
        key = (str(row.get("method")), str(row.get("task")), str(row.get("backend")), row.get("model"))
        groups.setdefault(key, []).append(row)

    mean_fields = (
        "precision", "recall", "f1", "coverage", "unsupported_rate", "ambiguous_rate",
        "human_edits_required", "evidence_grounded_rate", "requires_review_rate",
        "contextual_inference_rate", "field_contextual_inference_rate",
    )
    out: list[dict[str, Any]] = []
    for (method, task, backend, model), group in sorted(groups.items()):
        agg: dict[str, Any] = {
            "method": method,
            "task": task,
            "backend": backend,
            "model": model,
            "examples": len(group),
            "aggregation": "macro_mean_over_examples; micro_from_pooled_counts_when_available",
        }
        for field in mean_fields:
            vals = [float(row[field]) for row in group if isinstance(row.get(field), (int, float))]
            agg[field] = round(mean(vals), 6) if vals else None

        # Explicit aliases make the summary self-describing while keeping the
        # original keys stable for existing notebooks/app code.
        agg["precision_macro"] = agg.get("precision")
        agg["recall_macro"] = agg.get("recall")
        agg["f1_macro"] = agg.get("f1")
        agg["coverage_macro"] = agg.get("coverage")
        agg.update(_micro_scores(group))

        latencies = [float(row["latency_ms"]) for row in group if isinstance(row.get("latency_ms"), (int, float))]
        costs = [float(row["cost_usd_est"]) for row in group if isinstance(row.get("cost_usd_est"), (int, float))]
        agg["latency_ms"] = round(mean(latencies), 6) if latencies else None
        agg["latency_ms_mean_per_example"] = agg["latency_ms"]
        agg["latency_ms_total"] = round(sum(latencies), 6) if latencies else None
        agg["cost_usd_est"] = round(mean(costs), 8) if costs else None
        agg["cost_usd_est_mean_per_example"] = agg["cost_usd_est"]
        agg["cost_usd_est_total"] = round(sum(costs), 8) if costs else None
        out.append(agg)
    return out
