import pytest

from spatio_textual.journey_benchmark import (
    aggregate_journey_benchmark_rows,
    journey_benchmark_row,
)


def _journey(start, end, a, b, *, transport=None, reason=None, grounded=True, review=False, inferred=False):
    statuses = {
        "start_location": "contextual_inference" if inferred and start else ("explicit" if start else "missing"),
        "end_location": "explicit" if end else "missing",
        "transport_mode": "explicit" if transport else "missing",
        "date": "missing",
        "journey_reason": "explicit" if reason else "missing",
    }
    return {
        "start_location": start,
        "end_location": end,
        "transport_mode": transport,
        "date": None,
        "journey_reason": reason,
        "evidence_quote": "x" if grounded else None,
        "evidence_start_char": a if grounded else None,
        "evidence_end_char": b if grounded else None,
        "evidence_grounded": grounded,
        "explicit_or_inferred": statuses,
        "requires_review": review or inferred or not grounded,
    }


def test_journey_benchmark_row_keeps_detection_field_and_audit_metrics_separate():
    ref = [_journey("Leeds", "York", 0, 20, transport="train", reason="visit")]
    pred = [_journey("Leeds", "York", 0, 20, transport="by train", reason="visit")]
    row = journey_benchmark_row(
        example_id="ex1",
        predicted=pred,
        reference=ref,
        telemetry=[{
            "latency_ms": 100.0,
            "input_tokens_est": 20,
            "output_tokens_est": 10,
            "cost_usd_est": None,
            "success": True,
        }],
        provider="test",
        model="fixture",
    )
    assert row["f1"] == 1.0
    assert row["field_scoring"]["totals"]["precision"] == 1.0
    assert row["evidence_grounded_rate"] == 1.0
    assert row["telemetry_summary"]["latency_ms_total"] == 100.0


def test_pooled_summary_does_not_reward_empty_empty_examples():
    matched = journey_benchmark_row(
        example_id="positive",
        predicted=[_journey("A", "B", 0, 10)],
        reference=[_journey("A", "B", 0, 10)],
    )
    empty = journey_benchmark_row(
        example_id="negative",
        predicted=[],
        reference=[],
    )
    summary = aggregate_journey_benchmark_rows([matched, empty])
    assert summary["detection"]["tp"] == 1
    assert summary["detection"]["fp"] == 0
    assert summary["detection"]["fn"] == 0
    assert summary["detection"]["f1"] == 1.0
    assert summary["reference_journeys"] == 1
    assert summary["predicted_journeys"] == 1


def test_pooled_summary_counts_false_positive_on_no_journey_passage():
    positive = journey_benchmark_row(
        example_id="positive",
        predicted=[_journey("A", "B", 0, 10)],
        reference=[_journey("A", "B", 0, 10)],
    )
    negative_fp = journey_benchmark_row(
        example_id="negative",
        predicted=[_journey("X", "Y", 0, 10, grounded=False)],
        reference=[],
    )
    summary = aggregate_journey_benchmark_rows([positive, negative_fp])
    assert summary["detection"]["tp"] == 1
    assert summary["detection"]["fp"] == 1
    assert summary["detection"]["precision"] == 0.5
    assert summary["audit"]["evidence_grounded_rate"] == 0.5
    assert summary["audit"]["requires_review_rate"] == 0.5


def test_both_missing_fields_do_not_inflate_pooled_field_precision_recall():
    ref = [_journey("A", "B", 0, 10, reason="visit")]
    pred = [_journey("A", "B", 0, 10)]
    row = journey_benchmark_row(example_id="ex", predicted=pred, reference=ref)
    summary = aggregate_journey_benchmark_rows([row])
    totals = summary["fields"]["totals"]
    assert totals["correct"] == 2
    assert totals["missing_prediction"] == 1
    assert totals["both_missing"] == 2
    assert totals["precision"] == 1.0
    assert totals["recall"] == 0.666667


def test_journey_benchmark_rejects_backend_failure():
    with pytest.raises(ValueError, match="backend failure.*failed"):
        journey_benchmark_row(
            example_id="failed", predicted=[], reference=[],
            telemetry=[{"success": False, "error": "timeout"}],
        )


def test_pooled_telemetry_preserves_unknown_measurements():
    known = journey_benchmark_row(
        example_id="known", predicted=[], reference=[],
        telemetry=[{
            "success": True, "latency_ms": 10.0,
            "input_tokens_est": 4, "output_tokens_est": 2,
        }],
    )
    unknown = journey_benchmark_row(
        example_id="unknown", predicted=[], reference=[],
        telemetry=[{
            "success": True, "latency_ms": None,
            "input_tokens_est": None, "output_tokens_est": None,
        }],
    )
    summary = aggregate_journey_benchmark_rows([known, unknown])
    assert summary["telemetry"]["calls"] == 2
    assert summary["telemetry"]["latency_ms_known_calls"] == 1
    assert summary["telemetry"]["latency_ms_total"] is None
    assert summary["telemetry"]["latency_ms_mean_per_call"] is None
    assert summary["telemetry"]["input_tokens_est_total"] is None
    assert summary["telemetry"]["output_tokens_est_total"] is None
