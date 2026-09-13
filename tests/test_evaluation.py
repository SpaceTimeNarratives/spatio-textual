from spatio_textual.evaluation import (
    harmonize_ner_entities,
    reference_spans_for_ner,
    supported_reference_fraction,
)


def test_harmonise_common_place_labels_without_counting_person_org_as_spatial_errors():
    entities = [
        {"text": "London", "label": "GPE", "start_char": 0, "end_char": 6},
        {"text": "river", "label": "GEONOUN", "start_char": 10, "end_char": 15},
        {"text": "Ada", "label": "PERSON", "start_char": 20, "end_char": 23},
    ]
    out = harmonize_ner_entities(entities)
    assert [(r["text"], r["label"], r["model_label"]) for r in out] == [
        ("London", "TOPONYM", "GPE"),
        ("river", "GEONOUN", "GEONOUN"),
    ]


def test_harmonise_hf_location_label():
    out = harmonize_ner_entities([
        {"text": "Amsterdam", "label": "LOC", "start_char": 3, "end_char": 12}
    ])
    assert out[0]["label"] == "TOPONYM"
    assert out[0]["model_label"] == "LOC"


def test_temporal_labels_are_opt_in_for_ner_comparison():
    entity = {"text": "1938", "label": "DATE", "start_char": 0, "end_char": 4}
    assert harmonize_ner_entities([entity]) == []
    assert harmonize_ner_entities([entity], include_temporal=True)[0]["label"] == "TIME"


def test_reference_selection_distinguishes_named_place_from_hybrid_task():
    record = {
        "spans": [
            {"label": "TOPONYM", "text": "London"},
            {"label": "GEONOUN", "text": "river"},
            {"label": "DISTANCE", "text": "six miles"},
        ]
    }
    pure = reference_spans_for_ner(record)
    hybrid = reference_spans_for_ner(record, include_geonouns=True)
    assert [s["label"] for s in pure] == ["TOPONYM"]
    assert [s["label"] for s in hybrid] == ["TOPONYM", "GEONOUN"]


def test_ontology_ceiling_is_explicit_not_misreported_as_model_recall():
    record = {
        "spans": [
            {"label": "TOPONYM"},
            {"label": "GEONOUN"},
            {"label": "DISTANCE"},
            {"label": "DIRECTION"},
        ]
    }
    result = supported_reference_fraction(record, {"TOPONYM"})
    assert result["reference_total"] == 4
    assert result["ontology_supported"] == 1
    assert result["max_span_recall_from_ontology"] == 0.25
