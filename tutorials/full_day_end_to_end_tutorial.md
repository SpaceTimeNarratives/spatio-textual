# Full-day tutorial: spatio-textual annotation platform

## Recommended tutorial setup

Use the lightweight installation for teaching:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip wheel
python -m pip install -r requirements-lite.txt
streamlit run app.py
```

Avoid `python -m pip install -U pip | pip install ...`; the pipe sends stdout from the first command into the second command and is not a proper sequential install. Use `&&` or separate commands.

## Learning goals

By the end of the day, participants should be able to:

1. Segment testimonies into Q/A-aware turns.
2. Annotate spatial entities with place categories.
3. Link named places to latitude/longitude where possible.
4. Understand unresolved and ambiguous place-name review.
5. Compare spaCy and Hugging Face NER models.
6. Run MoE annotation and adjudicate disagreements.
7. Add sentiment and emotion distributions.
8. Extract narrator-centred actions/events.
9. Inspect telemetry and export audit-ready data.
10. Generate GeoJSON and co-occurrence outputs for later visualisation.

## Schedule

### 09:30 to 10:15: Conceptual overview

Introduce spatial humanities, testimony segmentation, annotation uncertainty, entity linking, affect analysis and telemetry.

### 10:15 to 11:00: Installing and launching

Use `requirements-lite.txt` and `streamlit run app.py`. Explain why this mode uses `en_core_web_sm` for speed.

### 11:00 to 12:00: Spatial entity annotation

Run the sample testimony. Inspect `entities`, place categories, lat/lon fields, unresolved places and `requires_review`.

### 12:00 to 12:45: Segmentation audit

Switch between Q/A-aware segmentation and sentence-safe character-budget segmentation. Inspect `segStartChar`, `segEndChar`, `turnId`, `qaPairId`, `isQuestion` and `isAnswer`.

### 13:45 to 14:45: Model comparison

Compare:

- `spacy:en_core_web_sm`
- `spacy:en_core_web_trf`
- `hf:dslim/bert-base-NER`
- `hf:dbmdz/bert-large-cased-finetuned-conll03-english`

Use a short text first. Explain that transformer models are stronger but slower and need more memory.

### 14:45 to 15:30: MoE and adjudication

Enable MoE. Show how disagreements are flagged for human review before export.

### 15:30 to 16:15: Sentiment, emotion and events

Run rule-based sentiment/emotion first. Then discuss HF/LLM options. Inspect distributions and mixed labels. Review `event_data` as narrator-centred action/experience extraction rather than a simple verb list.

### 16:15 to 17:00: Telemetry, export and next steps

Inspect telemetry. Export JSONL, CSV and CoNLL/BIO. Generate GeoJSON and co-occurrence edge lists. Discuss how the analysis and visualisation layer should build on the tidied annotation schema.

## Instructor note

For a smooth workshop, pre-build a Docker image or provide a prepared environment for the transformer/LLM section. Keep the first half of the day on the lightweight rule and spaCy-small pipeline.
