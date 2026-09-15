from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path

import spacy

import spatio_textual.cli as cli
import spatio_textual.emotion as emotion_module
import spatio_textual.sentiment as sentiment_module
from spatio_textual.emotion import EmotionAnalyzer
from spatio_textual.model_registry import TUTORIAL_NER_MODEL
from spatio_textual.sentiment import SentimentAnalyzer
from spatio_textual.utils import Annotator, serialize_annotations


ROOT = Path(__file__).resolve().parents[1]


def test_lightweight_app_defaults_to_its_installed_spacy_model():
    requirements = (ROOT / "requirements-lite.txt").read_text(encoding="utf-8")
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert TUTORIAL_NER_MODEL == "spacy:en_core_web_sm"
    assert "en_core_web_sm" in requirements
    assert "index=model_options.index(TUTORIAL_NER_MODEL)" in app_source


def test_release_metadata_is_canonical_and_consistent():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert 'version = "0.4.1"' in pyproject
    assert 'requires = ["setuptools>=77", "wheel"]' in pyproject
    assert 'license = "MIT"' in pyproject
    assert "GNU General Public License" not in pyproject
    assert "https://github.com/SpaceTimeNarratives/spatio-textual" in pyproject
    assert "version=" not in setup_source
    assert "install_requires" not in setup_source
    assert "GNU General Public License" not in setup_source


def test_llm_affect_uses_provider_default_when_model_is_omitted(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    class _Client:
        def __init__(self, provider, model):
            calls.append((provider, model))

        def classify_json(self, task, text, labels, instructions):
            return {
                "label": "mixed",
                "distribution": {label: 0.0 for label in labels},
                "telemetry": {"success": True},
            }

    monkeypatch.setattr(sentiment_module, "LLMClient", _Client)
    monkeypatch.setattr(emotion_module, "LLMClient", _Client)

    SentimentAnalyzer(backend="llm", provider="anthropic").predict(["text"])
    EmotionAnalyzer(backend="llm", provider="groq").predict(["text"])

    assert calls == [("anthropic", None), ("groq", None)]


def test_structured_segments_preserve_text_and_identifiers():
    raw = [
        {
            "text": "We moved to Amsterdam.",
            "file": "source/transcript.txt",
            "fileId": "doc-7",
            "segId": 42,
            "segCount": 80,
            "role": "answer",
            "turnId": 12,
            "qaPairId": 6,
            "isAnswer": True,
        }
    ]
    segments = cli._normalise_input_segments(raw)
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    records = Annotator(nlp, link_places=False).annotate_texts(
        segments,
        file_id="segments",
        include_text=True,
        include_events=False,
    )

    assert records[0]["text"] == raw[0]["text"]
    assert records[0]["file"] == "source/transcript.txt"
    assert records[0]["fileId"] == "doc-7"
    assert records[0]["segId"] == 42
    assert records[0]["segCount"] == 80
    assert records[0]["role"] == "answer"
    assert records[0]["turnId"] == 12
    assert records[0]["qaPairId"] == 6
    assert records[0]["isAnswer"] is True


def test_segments_json_rejects_records_without_text():
    try:
        cli._normalise_input_segments([{"fileId": "missing-text"}])
    except ValueError as exc:
        assert "string 'text' field" in str(exc)
    else:
        raise AssertionError("Invalid structured segment was accepted")


def test_serialization_supports_all_cli_output_formats():
    records = [{"fileId": "doc", "segId": 1, "text": "Amsterdam"}]

    assert json.loads(serialize_annotations(records, "json")) == records
    assert json.loads(serialize_annotations(records, "jsonl")) == records[0]
    csv_payload = serialize_annotations(records, "csv")
    tsv_payload = serialize_annotations(records, "tsv")
    assert "\r" not in csv_payload
    assert "\r" not in tsv_payload
    assert list(csv.DictReader(StringIO(csv_payload)))[0]["fileId"] == "doc"
    assert list(csv.DictReader(StringIO(tsv_payload), delimiter="\t"))[0]["fileId"] == "doc"


def test_cli_stdout_honours_jsonl(tmp_path, capsys):
    input_path = tmp_path / "segments.json"
    input_path.write_text(json.dumps(["Amsterdam was home."]), encoding="utf-8")

    result = cli.main(
        [
            "--segments-json",
            str(input_path),
            "--ner-model",
            "spacy:model-that-does-not-exist",
            "--no-link-places",
            "--no-events",
            "--output",
            "-",
            "--output-format",
            "jsonl",
        ]
    )

    output_lines = capsys.readouterr().out.splitlines()
    assert result == 0
    assert len(output_lines) == 1
    assert json.loads(output_lines[0])["text"] == "Amsterdam was home."
