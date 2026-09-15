from spatio_textual import emotion as emotion_module
from spatio_textual import sentiment as sentiment_module
from spatio_textual.emotion import EmotionAnalyzer
from spatio_textual.sentiment import SentimentAnalyzer


def test_sentiment_hf_revision_is_forwarded_and_recorded(monkeypatch):
    seen = {}

    def fake_loader(model_name, model_revision=None):
        seen["model"] = model_name
        seen["revision"] = model_revision

        def classify(_text):
            return [
                {"label": "positive", "score": 0.8},
                {"label": "neutral", "score": 0.15},
                {"label": "negative", "score": 0.05},
            ]

        return classify

    monkeypatch.setattr(sentiment_module, "_hf_sentiment", fake_loader)
    analyzer = SentimentAnalyzer(
        backend="hf",
        model_name="example/sentiment",
        model_revision="abc123",
    )
    result = analyzer.predict(["A short example."])[0]

    assert seen == {"model": "example/sentiment", "revision": "abc123"}
    assert result["label"] == "positive"
    assert result["telemetry"]["model"] == "example/sentiment"
    assert result["telemetry"]["model_revision"] == "abc123"


def test_emotion_hf_revision_is_forwarded_and_recorded(monkeypatch):
    seen = {}

    def fake_loader(model_name, model_revision=None):
        seen["model"] = model_name
        seen["revision"] = model_revision

        def classify(_text):
            return [
                {"label": "joy", "score": 0.7},
                {"label": "neutral", "score": 0.2},
                {"label": "sadness", "score": 0.1},
            ]

        return classify

    monkeypatch.setattr(emotion_module, "_hf_emotion", fake_loader)
    analyzer = EmotionAnalyzer(
        backend="hf",
        model_name="example/emotion",
        model_revision="def456",
    )
    result = analyzer.predict(["A short example."])[0]

    assert seen == {"model": "example/emotion", "revision": "def456"}
    assert result["label"] == "Joy"
    assert result["telemetry"]["model"] == "example/emotion"
    assert result["telemetry"]["model_revision"] == "def456"


def test_rule_affect_telemetry_has_explicit_null_revision():
    sentiment = SentimentAnalyzer("rule").predict(["A neutral sentence."])[0]
    emotion = EmotionAnalyzer("rule").predict(["A neutral sentence."])[0]

    assert sentiment["telemetry"]["model_revision"] is None
    assert emotion["telemetry"]["model_revision"] is None


def test_rule_analyzers_expose_public_lexical_explanations():
    sentiment = SentimentAnalyzer("rule").explain("We felt safe but afraid.")
    emotion = EmotionAnalyzer("rule").explain("We felt safe but afraid.")

    assert sentiment["positive_terms"] == ["safe"]
    assert sentiment["negative_terms"] == ["afraid"]
    assert emotion["matched_terms"]["Joy"] == ["safe"]
    assert emotion["matched_terms"]["Fear"] == ["afraid"]


def test_rule_explanations_do_not_claim_to_explain_model_backends():
    for analyzer in (SentimentAnalyzer("hf"), EmotionAnalyzer("llm")):
        try:
            analyzer.explain("A passage")
        except ValueError as exc:
            assert "rule" in str(exc)
        else:
            raise AssertionError("Non-rule backend unexpectedly returned lexical cues")
