from spatio_textual.formats import bio_to_entities, entities_to_bio
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


def test_affect_rules():
    sent = SentimentAnalyzer().predict(["I was afraid but later felt relief"])[0]
    emo = EmotionAnalyzer().predict(["I was afraid"])[0]
    assert sent["label"] in {"positive", "negative", "neutral"}
    assert emo["label"] == "Fear"


def test_annotator_fallback():
    nlp = load_spacy_model("model_that_does_not_exist")
    ann = Annotator(nlp)
    rec = ann.annotate("Amsterdam was cold.", include_text=True)
    assert "entities" in rec
