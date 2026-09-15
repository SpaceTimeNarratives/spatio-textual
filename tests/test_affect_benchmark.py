import pytest

from spatio_textual.affect_benchmark import evaluate_affect_predictions, score_multilabel, score_single_label
from spatio_textual.affect_rules import classify_affect_rule


def test_rule_does_not_encode_domain_terms_as_affect():
    pred = classify_affect_rule("The former camp boundary appears on the map beside the road.")
    assert pred["sentiment_label"] == "neutral"
    assert pred["emotion_labels"] == []


def test_rule_handles_simple_negation():
    pred = classify_affect_rule("Mira was not afraid of the short tunnel.")
    assert pred["sentiment_label"] == "neutral"
    assert pred["emotion_labels"] == []


def test_rule_supports_multilabel_explicit_affect():
    pred = classify_affect_rule("I was frightened by the delay but grateful when the guide returned.")
    assert pred["sentiment_label"] == "mixed"
    assert pred["emotion_labels"] == ["fear", "gratitude"]
    assert pred["evidence_quote"] is not None


def test_single_label_macro_scoring():
    result = score_single_label(
        ["positive", "negative", "neutral", "mixed"],
        ["positive", "negative", "neutral", "neutral"],
    )
    assert result["accuracy"] == 0.75
    assert result["per_label"]["positive"]["f1"] == 1.0
    assert result["per_label"]["mixed"]["recall"] == 0.0


def test_multilabel_micro_scoring():
    result = score_multilabel(
        [["fear", "anxiety"], ["joy"], []],
        [["fear"], ["joy", "surprise"], []],
    )
    assert result["micro"]["tp"] == 2
    assert result["micro"]["fp"] == 1
    assert result["micro"]["fn"] == 1


def test_affect_evaluation_records_representational_ceiling_and_grounding():
    refs = [
        {
            "example_id": "a1",
            "text": "I was grateful for the directions.",
            "sentiment_label": "positive",
            "emotion_labels": ["gratitude"],
        },
        {
            "example_id": "a2",
            "text": "The road crosses the river.",
            "sentiment_label": "neutral",
            "emotion_labels": [],
        },
    ]
    preds = [
        {
            "example_id": "a1",
            "sentiment_label": "positive",
            "emotion_labels": ["gratitude"],
            "evidence_quote": "grateful",
            "explicit_or_inferred": "explicit",
            "requires_review": False,
        },
        {
            "example_id": "a2",
            "sentiment_label": "neutral",
            "emotion_labels": [],
            "evidence_quote": None,
            "explicit_or_inferred": "none",
            "requires_review": False,
        },
    ]
    result = evaluate_affect_predictions(refs, preds, representable_emotion_labels=["fear", "joy"])
    assert result["evidence_grounded_rate"] == 1.0
    assert result["emotion_representational_ceiling"] == 0.0


def test_affect_evaluation_rejects_backend_failures():
    refs = [{
        "example_id": "failed", "text": "A plain sentence.",
        "sentiment_label": "neutral", "emotion_labels": [],
    }]
    preds = [{
        "example_id": "failed", "sentiment_label": "neutral", "emotion_labels": [],
        "backend_error": True,
        "telemetry": [{"success": False, "error": "provider unavailable"}],
    }]
    with pytest.raises(ValueError, match="backend failures.*failed"):
        evaluate_affect_predictions(refs, preds)


def test_affect_evaluation_rejects_malformed_responses():
    refs = [{
        "example_id": "bad", "text": "A plain sentence.",
        "sentiment_label": "neutral", "emotion_labels": [],
    }]
    preds = [{
        "example_id": "bad", "sentiment_label": "neutral", "emotion_labels": [],
        "invalid_response": True,
    }]
    with pytest.raises(ValueError, match="malformed responses.*bad"):
        evaluate_affect_predictions(refs, preds)
