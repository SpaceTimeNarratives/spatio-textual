from spatio_textual.journey_evaluation import (
    candidate_journey_match,
    evaluate_journeys,
    evidence_iou,
    match_journeys,
    normalize_journey_value,
    score_matched_journey_fields,
)


def _journey(start, end, evidence_start, evidence_end, *, transport=None, date=None, reason=None, quote=None):
    return {
        "start_location": start,
        "end_location": end,
        "transport_mode": transport,
        "date": date,
        "journey_reason": reason,
        "evidence_start_char": evidence_start,
        "evidence_end_char": evidence_end,
        "evidence_quote": quote,
    }


def test_transport_normalization_is_transparent_and_bounded():
    assert normalize_journey_value("on foot", "transport_mode") == "foot"
    assert normalize_journey_value("Walking", "transport_mode") == "foot"
    assert normalize_journey_value("plane", "transport_mode") == "air"
    assert normalize_journey_value("horse cart", "transport_mode") == "horse cart"


def test_evidence_iou_uses_source_offsets():
    ref = _journey("Oxford", "Reading", 10, 30)
    pred = _journey("Oxford", "Reading", 20, 40)
    assert evidence_iou(ref, pred) == 10 / 30


def test_candidate_matches_all_reference_endpoints_without_same_evidence_span():
    ref = _journey("Oxford", "Reading", 0, 20)
    pred = _journey("oxford", "reading", 50, 70)
    detail = candidate_journey_match(ref, pred)
    assert detail["eligible"] is True
    assert detail["eligibility_reason"] == "all_reference_endpoints_match"
    assert detail["endpoint_matches"] == 2


def test_evidence_overlap_can_match_partially_wrong_record_for_field_scoring():
    ref = _journey("Oxford", "Reading", 0, 100)
    pred = _journey("Oxford", "London", 20, 90)
    detail = candidate_journey_match(ref, pred)
    assert detail["eligible"] is True
    assert detail["eligibility_reason"] == "evidence_overlap_plus_endpoint"
    assert detail["endpoint_matches"] == 1


def test_unrelated_destination_is_not_match():
    ref = _journey("Oxford", "Reading", 0, 20)
    pred = _journey("Leeds", "York", 0, 20)
    assert candidate_journey_match(ref, pred)["eligible"] is False


def test_evidence_only_reference_matches_by_grounded_span():
    ref = _journey(None, None, 10, 40, quote="the same movement evidence")
    pred = _journey(None, None, 10, 40, quote="the same movement evidence")
    detail = candidate_journey_match(ref, pred)
    assert detail["eligible"] is True
    assert detail["eligibility_reason"] == "evidence_overlap_no_endpoints"
    result = match_journeys([pred], [ref])
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0


def test_matching_is_one_to_one_and_reports_unmatched_records():
    refs = [
        _journey("A", "B", 0, 20),
        _journey("B", "C", 30, 50),
    ]
    preds = [
        _journey("A", "B", 0, 20),
        _journey("X", "Y", 60, 80),
    ]
    result = match_journeys(preds, refs)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["unmatched_pred_indices"] == [1]
    assert result["unmatched_ref_indices"] == [1]


def test_field_scoring_does_not_reward_both_missing():
    refs = [{
        "start_location": "Oxford",
        "end_location": "Reading",
        "transport_mode": None,
        "date": None,
        "journey_reason": "meeting",
    }]
    preds = [{
        "start_location": "Oxford",
        "end_location": "Reading",
        "transport_mode": "train",
        "date": None,
        "journey_reason": "conference",
    }]
    matches = [{"pred_index": 0, "ref_index": 0}]
    score = score_matched_journey_fields(preds, refs, matches)
    totals = score["totals"]
    assert totals["correct"] == 2
    assert totals["unsupported_prediction"] == 1
    assert totals["mismatch"] == 1
    assert totals["both_missing"] == 1
    assert totals["predicted_nonmissing"] == 4
    assert totals["reference_nonmissing"] == 3
    assert totals["precision"] == 0.5
    assert totals["recall"] == 0.666667
    assert totals["unsupported_field_rate"] == 0.5


def test_evaluate_journeys_combines_record_and_field_metrics():
    refs = [_journey("Leeds", "Manchester", 0, 60, transport="train", date="Monday", reason="meeting")]
    preds = [_journey("Leeds", "Manchester", 0, 60, transport="by train", date="Monday", reason="meeting")]
    result = evaluate_journeys(preds, refs)
    assert result["matching"]["f1"] == 1.0
    assert result["field_scoring"]["totals"]["precision"] == 1.0
    assert result["field_scoring"]["totals"]["recall"] == 1.0
