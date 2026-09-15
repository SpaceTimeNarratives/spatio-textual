from spatio_textual.gold import find_span, score_span_annotations


def test_find_span_uses_one_based_occurrence_and_exact_offsets():
    text = "London then London"
    second = find_span(text, "London", "TOPONYM", occurrence=2, layer="entity")
    assert second["start_char"] == 12
    assert second["end_char"] == 18
    assert text[second["start_char"]:second["end_char"]] == second["text"]


def test_exact_span_scoring_is_perfect_without_project_fixture():
    text = "From London to York."
    spans = [
        find_span(text, "London", "TOPONYM", layer="entity"),
        find_span(text, "York", "TOPONYM", layer="entity"),
    ]
    score = score_span_annotations(spans, spans, match="exact")
    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
    assert score["f1"] == 1.0


def test_overlap_span_scoring_accepts_partial_boundary_overlap():
    text = "about six miles distant"
    reference = [find_span(text, "about six miles distant", "DISTANCE", layer="spatial_cue")]
    predicted = [find_span(text, "six miles", "DISTANCE", layer="spatial_cue")]
    exact = score_span_annotations(predicted, reference, match="exact")
    overlap = score_span_annotations(predicted, reference, match="overlap")
    assert exact["f1"] == 0.0
    assert overlap["f1"] == 1.0
    assert overlap["matches"][0]["overlap_iou"] < 1.0
