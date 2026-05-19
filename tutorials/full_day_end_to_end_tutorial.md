# Full-day end-to-end tutorial: spatio-textual annotation platform

**Preferred delivery:** Streamlit on Hugging Face Spaces, with Colab as fallback.

## Learning outcomes

By the end of the day, participants will be able to:

1. Upload or paste testimony-style text into the web app.
2. Segment testimony transcripts into Q/A-aware turns.
3. Annotate spatial entities and inspect place classifications.
4. Run rule-based sentiment and emotion analysis over segments.
5. Use MoE-style adjudication to inspect consensus and disagreement.
6. Export JSONL, JSON, CSV/TSV, BIO/CoNLL and visualisation artefacts.
7. Run the same workflow from Python and the CLI for reproducibility.

## Suggested timetable

| Time | Session | Activities | Output |
|---|---|---|---|
| 09:30-10:00 | Setup and orientation | Open the HF Space, duplicate it if needed, inspect the sample text | Working personal Space |
| 10:00-10:45 | Spatial entities | Run entity annotation, inspect entity tables, explain labels and place types | Entity table |
| 10:45-11:00 | Break |  |  |
| 11:00-12:00 | Testimony segmentation | Compare raw chunking with Q/A-aware turns, discuss interviewer and witness roles | Turn-level records |
| 12:00-12:30 | Affect analysis | Run sentiment and emotion, inspect scores and labels | Affect-enhanced JSON |
| 12:30-13:30 | Lunch |  |  |
| 13:30-14:15 | MoE adjudication | Compare `spaCy+ruler` and `spaCy`, inspect agreement and disagreement | Consensus entities |
| 14:15-15:00 | Visualisation | Build co-occurrence edges and GeoJSON, discuss geocoding options | Edge list and GeoJSON |
| 15:00-15:15 | Break |  |  |
| 15:15-16:00 | Reproducible CLI workflow | Run the CLI over a folder of texts with `--workers` and `--tqdm` | JSONL corpus output |
| 16:00-16:45 | Python workflow | Load JSONL with pandas, convert entities to BIO/CoNLL, save outputs | DataFrame and CoNLL |
| 16:45-17:00 | Wrap-up | Discuss responsible use, provenance, human review and next steps | Action plan |

## Facilitator setup

```bash
git clone https://github.com/SpaceTimeNarratives/spatio-textual.git
cd spatio-textual
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[app,dev]
python -m spacy download en_core_web_sm
streamlit run app.py
```

## Exercise 1: web annotation

1. Open the Streamlit app.
2. Paste the sample testimony from `example-texts/sample_testimony.txt`.
3. Keep Q/A-aware segmentation switched on.
4. Run annotation.
5. Inspect the Records and Entities tabs.

Discussion prompts:

- Which entities are named places and which are spatial nouns?
- Which entities require human correction?
- What additional project-specific EntityRuler terms should be added?

## Exercise 2: affect analysis

1. Enable Sentiment, Emotion and Interpretation.
2. Re-run annotation.
3. Compare sentiment and emotion labels across interviewer and witness turns.
4. Discuss why rule-based affect labels should be treated as weak signals.

## Exercise 3: MoE adjudication

1. Select both `spaCy+ruler` and `spaCy`.
2. Run annotation.
3. Open the Adjudication tab.
4. Inspect consensus entities and disagreement items.

Discussion prompts:

- Which disagreements are useful for human review?
- What should be the acceptance threshold for different projects?
- How would LLM or HF model outputs be added as additional experts?

## Exercise 4: CLI workflow

```bash
spatio-textual \
  -i example-texts/ --glob "*.txt" --tqdm \
  --testimony --sentiment rule --emotion rule --interpret --verbs \
  --workers 2 --chunksize 8 \
  -o out/tutorial_records.jsonl --output-format jsonl
```

Inspect the output:

```python
from spatio_textual.utils import load_annotations

df = load_annotations("out/tutorial_records.jsonl")
df[["fileId", "segId", "role", "sentiment_label", "emotion_label", "themes"]]
```

## Exercise 5: BIO and CoNLL export

```python
from spatio_textual.formats import entities_to_conll
from spatio_textual.utils import load_annotations

df = load_annotations("out/tutorial_records.jsonl")
for _, row in df.iterrows():
    tokens = row["text"].split()
    conll = entities_to_conll(tokens, row["entities"], doc_id=f"{row['fileId']}_{row['segId']}")
    print(conll)
```

## Exercise 6: visualisation outputs

```bash
spatio-textual \
  -i example-texts/sample_testimony.txt --testimony --sentiment rule --emotion rule \
  --cooccurrence-out out/cooccurrence.tsv \
  --geojson-out out/places.geojson \
  -o out/records.jsonl --output-format jsonl
```

The GeoJSON file will only contain features for entities that already include coordinates or when a geocoder is supplied in Python.

## Responsible use notes

- Treat rule-based sentiment and emotion as triage signals, not final scholarly claims.
- Keep evidence text and segment IDs with every annotation.
- Use adjudication to prioritise human review rather than to replace it.
- Document all custom resources and prompt/model versions used in production annotation.
- For Holocaust testimony data, keep data governance, consent, access restrictions and survivor dignity central to the workflow.
