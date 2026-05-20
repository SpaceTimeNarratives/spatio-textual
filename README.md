# spatio-textual ✨

A Python package and Streamlit/HF Space platform for spatial textual annotation, testimony segmentation, entity linking, affect analysis, narrator-centred event extraction, telemetry and visualisation.

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

## Do I need to show Hugging Face YAML in the GitHub README?

No. Hugging Face Spaces reads configuration from the YAML block at the top of the README in the **Space repository**. Keep the main GitHub README clean. Use `hf_space/README.md` as the README for the Space repository, or use the included `Dockerfile` and copy the YAML only to the HF Space repo.

## Fast local tutorial install

Do **not** pipe the two pip commands. This is wrong and can hang or behave strangely:

```bash
python -m pip install -U pip | pip install -e ".[app,dev]"
```

Use one of these instead.

### Fast tutorial/dev mode

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

For the full-day tutorial, use `requirements-lite.txt` so participants start quickly. Demonstrate `en_core_web_trf` and HF models on a smaller sample or pre-built environment.

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
- export format.

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

The Dockerfile installs the fast tutorial model by default. For a heavier public demo, change it to use `requirements-transformers.txt`.

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
