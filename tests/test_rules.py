from pathlib import Path

from spatio_textual.rules import RuleGazetteerAnnotator, load_teaching_gazetteer


def _gazetteer(tmp_path: Path) -> Path:
    path = tmp_path / "gazetteer.csv"
    path.write_text("text,label\nPenrith,TOPONYM\nLondon,TOPONYM\n", encoding="utf-8")
    return path


def test_gazetteer_loader_accepts_minimal_text_label_csv(tmp_path):
    rows = load_teaching_gazetteer(_gazetteer(tmp_path))
    assert rows == [
        {"text": "Penrith", "label": "TOPONYM"},
        {"text": "London", "label": "TOPONYM"},
    ]


def test_rule_annotator_combines_gazetteer_and_generic_distance_rule(tmp_path):
    ann = RuleGazetteerAnnotator(
        gazetteer_path=_gazetteer(tmp_path),
        include_project_resources=False,
        link_places=False,
    )
    result = ann.annotate("From Penrith we travelled about six miles toward London.")
    found = {(row["text"], row["label"]) for row in result["spans"]}
    assert ("Penrith", "TOPONYM") in found
    assert ("London", "TOPONYM") in found
    assert ("travelled", "MOVEMENT_CUE") in found
    assert ("about six miles", "DISTANCE") in found


def test_rule_annotator_is_case_insensitive_by_default(tmp_path):
    ann = RuleGazetteerAnnotator(
        gazetteer_path=_gazetteer(tmp_path),
        include_project_resources=False,
        link_places=False,
    )
    result = ann.annotate("We left PENRITH for LONDON.")
    found = {(row["text"], row["label"]) for row in result["spans"]}
    assert ("PENRITH", "TOPONYM") in found
    assert ("LONDON", "TOPONYM") in found


def test_rule_telemetry_is_local_zero_cost_estimate(tmp_path):
    ann = RuleGazetteerAnnotator(
        gazetteer_path=_gazetteer(tmp_path),
        include_project_resources=False,
        link_places=False,
    )
    telemetry = ann.annotate("From Penrith to London.")["telemetry"][0]
    assert telemetry["backend"] == "rules"
    assert telemetry["provider"] == "local"
    assert telemetry["cost_usd_est"] == 0.0
    assert telemetry["success"] is True
