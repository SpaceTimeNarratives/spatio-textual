# spatio-textual ✨

A Python package and Streamlit/HF Space platform for spatial textual annotation, testimony segmentation, entity linking, affect analysis, narrator-centred event extraction, telemetry and visualisation.

## Repository scope

This repository contains the reusable Python package, its general applications,
documentation, examples and tests. Conference-specific notebooks, datasets,
experimental protocols and frozen results belong in their own project
repositories and depend on a tagged package release through the public API.

## What changed in v0.4.1

- Restored the legacy package-root annotation imports retained from v0.3.
- Preserved structured segment identifiers and format-correct CLI output.
- Corrected CoNLL/BIO alignment when model and export tokenizers differ.
- Let each LLM provider choose its own default model when none is supplied.
- Limited Streamlit NER and MoE choices to models supported by the installed runtime.
- Modernised canonical package metadata and expanded release regression coverage.

## What changed in v0.4

- Evidence-first structured spatial-span and journey extraction.
- Transparent rule and transformer baselines for spatial and affect analysis.
- Reference-data validation plus span, journey and affect evaluation utilities.
- Human-review operations with auditable correction histories.
- Reproducible provenance manifests and strict-schema OpenAI Responses support.
- Package APIs and identifiers are independent of any conference repository.

## What changed in v0.3

- Default high-quality NER option: `spacy:en_core_web_trf`.
- Fast tutorial option: `spacy:en_core_web_sm`.
- HF transformer NER comparison options:
  - `hf:dslim/bert-base-NER`
  - `hf:dbmdz/bert-large-cased-finetuned-conll03-english`
- Spatial entity linking/geocoding for physically mappable named places.
- Ambiguous/unresolved places are flagged for review.
- Sentiment returns a probability distribution over `positive`, `neutral`, `negative`; label becomes `mixed` when no label is clearly dominant.
- Emotion returns a probability distribution over `Neutral`, `Joy`, `Surprise`, `Sadness`, `Fear`, `Anger`, `Disgust`; label becomes `mixed` when no label is clearly dominant.
- Narrator-centred `event_data` extraction distinguishes first-person actions/experiences from a raw verb list.
- All annotation records include segmentation audit fields and telemetry: latency, model, backend, provider, estimated tokens and estimated cost.
- MoE/entity adjudication flags model disagreement for human correction before export.

## Installation

### Lightweight demo/development mode

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip wheel
python -m pip install -r requirements-lite.txt
streamlit run app.py
```

### High-quality transformer mode

```bash
python -m pip install -r requirements-transformers.txt
streamlit run app.py
```

### LLM/HF full mode

```bash
python -m pip install -r requirements-llm.txt
streamlit run app.py
```

For the complete teaching sequence, exercises and Colab notebooks, use the
[Spatial Humanities 2026 workshop repository](https://github.com/IgnatiusEzeani/spatial-humanities-2026).
It pins a released version of this package and keeps workshop-specific material
outside the reusable library.

## Local app

```bash
streamlit run app.py
```

The sidebar lets you choose:

- primary spatial NER model,
- MoE expert models,
- segmentation mode,
- place linking,
- sentiment backend,
- emotion backend,
- LLM provider,
- event/action extraction,
- JSON, JSONL, CSV and CoNLL/BIO downloads.

## CLI examples

Fast local run:

```bash
spatio-textual \
  -i example-texts --glob "*.txt" --testimony \
  --ner-model spacy:en_core_web_sm \
  --sentiment-backend rule --emotion-backend rule \
  --events --link-places --tqdm \
  -o out/annotations.jsonl --output-format jsonl
```

High-quality transformer spaCy run:

```bash
spatio-textual \
  -i example-texts --glob "*.txt" --testimony \
  --ner-model spacy:en_core_web_trf \
  --sentiment-backend rule --emotion-backend rule \
  --events --link-places \
  -o out/annotations.jsonl --output-format jsonl
```

HF NER comparison:

```bash
spatio-textual \
  -i example-texts/sample_testimony.txt \
  --ner-model hf:dslim/bert-base-NER \
  --sentiment-backend hf \
  --sentiment-model cardiffnlp/twitter-roberta-base-sentiment-latest \
  --emotion-backend hf \
  --emotion-model j-hartmann/emotion-english-distilroberta-base \
  -o out/hf_annotations.jsonl
```

MoE adjudication:

```bash
spatio-textual \
  -i example-texts/sample_testimony.txt \
  --moe-models spacy:en_core_web_trf spacy:en_core_web_sm hf:dslim/bert-base-NER \
  --moe-threshold 0.5 \
  --sentiment-backend rule --emotion-backend rule \
  -o out/moe_annotations.jsonl
```

## Hugging Face Space

Recommended: Docker Space.

1. Create a new Hugging Face Space with SDK `Docker`.
2. Copy this repo into the Space repository.
3. Copy `hf_space/README.md` to the Space repository root as `README.md`.
4. Push.

The Dockerfile installs the lightweight demo model by default. The app hides unavailable
transformer choices rather than silently running an uninstalled model. For a heavier
public demo, change the Dockerfile to use `requirements-transformers.txt`.

## Standard output fields

Each record includes:

```text
file, fileId, segId, segCount, segStartChar, segEndChar, segTextCharLength,
entities, verb_data, event_data, text, error,
role, turnId, qaPairId, isQuestion, isAnswer,
sentiment_label, sentiment_score, sentiment_distribution,
emotion_label, emotion_score, emotion_dist,
summary, interpretation, themes,
telemetry, requires_review, review_notes
```

## Telemetry

Each model step appends telemetry like:

```json
{
  "task": "spatial_entity_recognition",
  "backend": "spacy",
  "provider": "local",
  "model": "en_core_web_trf",
  "latency_ms": 123.4,
  "input_chars": 1000,
  "input_tokens_est": 250,
  "output_tokens_est": 40,
  "cost_usd_est": 0.0,
  "success": true,
  "error": null
}
```

Provider billing can be made exact later by plugging in provider-specific token counters and price tables. The current implementation gives a consistent offline estimate for audit and teaching.
