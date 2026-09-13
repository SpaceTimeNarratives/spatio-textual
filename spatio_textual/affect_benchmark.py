from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

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


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def score_single_label(
    reference: Iterable[str],
    predicted: Iterable[str],
    *,
    labels: tuple[str, ...] = SENTIMENT_LABELS,
) -> dict[str, Any]:
    refs = list(reference)
    preds = list(predicted)
    if len(refs) != len(preds):
        raise ValueError("reference and predicted lengths differ")
    per_label: dict[str, Any] = {}
    for label in labels:
        tp = sum(r == label and p == label for r, p in zip(refs, preds))
        fp = sum(r != label and p == label for r, p in zip(refs, preds))
        fn = sum(r == label and p != label for r, p in zip(refs, preds))
        per_label[label] = _prf(tp, fp, fn)
    macro_f1 = sum(float(per_label[label]["f1"]) for label in labels) / len(labels)
    accuracy = sum(r == p for r, p in zip(refs, preds)) / len(refs) if refs else 0.0
    return {
        "n": len(refs),
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "per_label": per_label,
        "reference_distribution": dict(Counter(refs)),
        "prediction_distribution": dict(Counter(preds)),
    }


def score_multilabel(
    reference: Iterable[Iterable[str]],
    predicted: Iterable[Iterable[str]],
    *,
    labels: tuple[str, ...] = EMOTION_LABELS,
) -> dict[str, Any]:
    refs = [set(x) for x in reference]
    preds = [set(x) for x in predicted]
    if len(refs) != len(preds):
        raise ValueError("reference and predicted lengths differ")
    allowed = set(labels)
    bad = sorted(set().union(*refs, *preds) - allowed) if refs or preds else []
    if bad:
        raise ValueError(f"unsupported multilabel values: {bad}")

    per_label: dict[str, Any] = {}
    micro_tp = micro_fp = micro_fn = 0
    for label in labels:
        tp = sum(label in r and label in p for r, p in zip(refs, preds))
        fp = sum(label not in r and label in p for r, p in zip(refs, preds))
        fn = sum(label in r and label not in p for r, p in zip(refs, preds))
        per_label[label] = _prf(tp, fp, fn)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
    macro_f1 = sum(float(per_label[label]["f1"]) for label in labels) / len(labels)
    exact_set_accuracy = sum(r == p for r, p in zip(refs, preds)) / len(refs) if refs else 0.0
    return {
        "n": len(refs),
        "exact_set_accuracy": round(exact_set_accuracy, 6),
        "micro": _prf(micro_tp, micro_fp, micro_fn),
        "macro_f1": round(macro_f1, 6),
        "per_label": per_label,
        "reference_positive_labels": sum(len(r) for r in refs),
        "predicted_positive_labels": sum(len(p) for p in preds),
    }


def _grounded_evidence(prediction: dict[str, Any], text: str) -> bool | None:
    quote = prediction.get("evidence_quote")
    if quote in (None, ""):
        return None
    return isinstance(quote, str) and quote in text


def evaluate_affect_predictions(
    references: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    representable_emotion_labels: Iterable[str] | None = None,
) -> dict[str, Any]:
    pred_by_id = {str(row["example_id"]): row for row in predictions}
    missing = [str(row["example_id"]) for row in references if str(row["example_id"]) not in pred_by_id]
    extras = sorted(set(pred_by_id) - {str(row["example_id"]) for row in references})
    if missing or extras:
        raise ValueError(f"prediction/reference ID mismatch: missing={missing}, extras={extras}")

    ordered_predictions = [pred_by_id[str(row["example_id"])] for row in references]
    failed_ids = [
        str(row["example_id"])
        for row in ordered_predictions
        if row.get("backend_error") is True
        or any(
            isinstance(item, dict) and item.get("success") is False
            for item in (
                row.get("telemetry")
                if isinstance(row.get("telemetry"), list)
                else [row.get("telemetry")]
            )
        )
    ]
    invalid_ids = [str(row["example_id"]) for row in ordered_predictions if row.get("invalid_response") is True]
    if failed_ids or invalid_ids:
        raise ValueError(
            "Cannot score invalid affect predictions; "
            f"backend failures={failed_ids}, malformed responses={invalid_ids}"
        )
    sentiment = score_single_label(
        [str(row["sentiment_label"]) for row in references],
        [str(row.get("sentiment_label") or "neutral") for row in ordered_predictions],
    )
    emotion = score_multilabel(
        [row.get("emotion_labels", []) for row in references],
        [row.get("emotion_labels", []) for row in ordered_predictions],
    )

    representable = set(representable_emotion_labels or EMOTION_LABELS)
    unsupported_reference_labels = 0
    total_reference_labels = 0
    for row in references:
        for label in row.get("emotion_labels", []):
            total_reference_labels += 1
            if label not in representable:
                unsupported_reference_labels += 1
    representational_ceiling = (
        1.0 - unsupported_reference_labels / total_reference_labels
        if total_reference_labels
        else 1.0
    )

    evidence_values = [
        _grounded_evidence(pred, str(ref["text"]))
        for ref, pred in zip(references, ordered_predictions)
    ]
    evidence_applicable = [v for v in evidence_values if v is not None]
    grounded_rate = (
        sum(bool(v) for v in evidence_applicable) / len(evidence_applicable)
        if evidence_applicable
        else None
    )
    review_rate = sum(bool(row.get("requires_review")) for row in ordered_predictions) / len(ordered_predictions) if ordered_predictions else 0.0
    inference_rate = sum(row.get("explicit_or_inferred") == "contextual_inference" for row in ordered_predictions) / len(ordered_predictions) if ordered_predictions else 0.0

    latencies = []
    for row in ordered_predictions:
        telemetry = row.get("telemetry")
        if isinstance(telemetry, dict) and isinstance(telemetry.get("latency_ms"), (int, float)):
            latencies.append(float(telemetry["latency_ms"]))
        elif isinstance(telemetry, list):
            latencies.extend(
                float(t["latency_ms"])
                for t in telemetry
                if isinstance(t, dict) and isinstance(t.get("latency_ms"), (int, float))
            )

    return {
        "schema_version": "spatio-textual-affect-evaluation-v1",
        "examples": len(references),
        "sentiment": sentiment,
        "emotion": emotion,
        "representable_emotion_labels": sorted(representable),
        "unsupported_reference_emotion_labels": unsupported_reference_labels,
        "reference_emotion_labels": total_reference_labels,
        "emotion_representational_ceiling": round(representational_ceiling, 6),
        "evidence_grounded_rate": None if grounded_rate is None else round(grounded_rate, 6),
        "evidence_applicable_examples": len(evidence_applicable),
        "review_required_rate": round(review_rate, 6),
        "contextual_inference_rate": round(inference_rate, 6),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else None,
    }
