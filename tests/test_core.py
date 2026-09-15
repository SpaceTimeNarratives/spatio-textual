import pytest

from spatio_textual.formats import bio_to_entities, entities_to_bio, entities_to_conll
from spatio_textual.qa import segment_testimony
from spatio_textual.sentiment import SentimentAnalyzer
from spatio_textual.emotion import EmotionAnalyzer
from spatio_textual.utils import Annotator, load_spacy_model, split_into_segments


def test_sentence_safe_chunking():
    text = "One sentence. Two sentence. Three sentence."
    chunks = split_into_segments(text, n_segments=2)
    assert len(chunks) == 2
    assert all(chunk.endswith(".") for chunk in chunks)


def test_qa_segmentation():
    text = "Q: Where were you?\nA: In Amsterdam."
    turns = segment_testimony(text)
    assert len(turns) == 2
    assert turns[0].is_question is True
    assert turns[1].is_answer is True
    assert turns[0].qa_pair_id == turns[1].qa_pair_id


def test_bio_roundtrip():
    tokens = ["Anne", "went", "to", "Amsterdam"]
    entities = [{"text": "Amsterdam", "label": "GPE", "start_token": 3, "end_token": 4}]
    tags = entities_to_bio(tokens, entities)
    assert tags == ["O", "O", "O", "B-GPE"]
    ents = bio_to_entities(tokens, tags)
    assert ents[0]["label"] == "GPE"


def test_conll_export_aligns_character_only_entities():
    text = "We moved from New York to London."
    tokens = text.split()
    entities = [
        {
            "text": "New York",
            "label": "GPE",
            "start_char": text.index("New York"),
            "end_char": text.index("New York") + len("New York"),
            "start_token": None,
            "end_token": None,
        }
    ]

    conll = entities_to_conll(tokens, entities, text=text)

    assert "New\tB-GPE" in conll
    assert "York\tI-GPE" in conll


def test_conll_export_prefers_character_offsets_across_tokenizers():
    text = "Hello, Amsterdam"
    tokens = text.split()
    entities = [
        {
            "text": "Amsterdam",
            "label": "GPE",
            "start_char": text.index("Amsterdam"),
            "end_char": len(text),
            "start_token": 2,
            "end_token": 3,
        }
    ]

    conll = entities_to_conll(tokens, entities, text=text)

    assert "Hello,\tO" in conll
    assert "Amsterdam\tB-GPE" in conll


def test_affect_rules():
    sent = SentimentAnalyzer().predict(["I was afraid but later felt relief"])[0]
    emo = EmotionAnalyzer().predict(["I was afraid"])[0]
    assert sent["label"] in {"positive", "negative", "neutral", "mixed"}
    assert set(sent["distribution"]) == {"positive", "neutral", "negative"}
    assert emo["label"] in {"Fear", "mixed"}
    assert "Fear" in emo["distribution"]


def test_annotator_fallback():
    nlp = load_spacy_model("model_that_does_not_exist")
    ann = Annotator(nlp)
    rec = ann.annotate("Amsterdam was cold.", include_text=True)
    assert "entities" in rec


@pytest.mark.parametrize(
    ("text", "expected_label"),
    [
        ("river", "GEONOUN"),
        ("[LAUGHS]", "NON-VERBAL"),
        ("mother", "FAMILY"),
        ("Auschwitz", "CAMP"),
        ("Amsterdam", "CITY"),
    ],
)
def test_packaged_resources_produce_their_configured_labels(text, expected_label):
    nlp = load_spacy_model("model_that_does_not_exist")

    assert [(ent.text, ent.label_) for ent in nlp(text).ents] == [(text, expected_label)]


def test_legacy_city_resource_filename_remains_supported(tmp_path):
    (tmp_path / "ambiguous_cities.txt").write_text("Legacyville\n", encoding="utf-8")

    nlp = load_spacy_model("model_that_does_not_exist", resources_dir=tmp_path)

    assert [(ent.text, ent.label_) for ent in nlp("Legacyville").ents] == [("Legacyville", "CITY")]


def test_annotation_telemetry_and_place_linking():
    nlp = load_spacy_model("model_that_does_not_exist")
    ann = Annotator(nlp)
    rec = ann.annotate("I travelled to Amsterdam.", include_text=True, include_verbs=True, include_events=True)
    assert "telemetry" in rec
    assert "event_data" in rec
