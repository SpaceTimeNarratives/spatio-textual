# spatio-textual ✨

A Python library for textual analysis for spatial and digital humanities research created as part of the [Spatial Narratives Project](https://spacetimenarratives.github.io/). It supports spatio-textual annotation, analysis and visualisation for digital humanities projects, with initial applications to:

- **Corpus of Lake District Writing (CLDW)**
- **Holocaust survivors' testimonies** (e.g., USC Shoah Foundation archives)

This release includes **sentence-safe chunking**, **list-of-texts input**, **file/segment IDs in output**, **robust saving of annotation (JSON/JSONL/TSV/CSV)**, **efficient loading of annotation (Pandas friendly)** and a **multiprocessing CLI with tqdm progress report** and a clean Python API.

Additional features:

* ✂️ **Testimony segmentation** (Q↔A-aware turns; roles: interviewer/witness)
* 😊 **Sentiment** (rule backend; LLM/HF hooks to follow)
* 😶‍🌫️ **Emotion** (Ekman-style labels via rule backend; LLM/HF hooks to follow)
* 🧠 **Interpretation** (summaries/themes; optional LLM hook)
* 🗺️📈 **Visualisation** (GeoJSON/Folium map; co-occurrence graphs)


---

## 🧭 What’s inside

**Core**

* spaCy model loader with optional **EntityRuler** patterns
* Place classification (COUNTRY, CITY, CONTINENT, CAMP, …)
* Sentence‑safe chunking (no broken sentences)
* Flexible annotation APIs for **single or multiple files** and **lists of texts**
* Robust saving in **JSON / JSONL / TSV / CSV** + a pandas‑friendly loader

**Testimony features**

* ✂️ Q↔A‑aware segmentation of testimonies (roles: `interviewer|witness|narration|unknown`), with `turnId`, `qaPairId`, `isQuestion`, `isAnswer`

**Affect analysis**

* 😊 Sentiment classification (rule backend; hooks for LLM/HF)
* 😶‍🌫️ Emotion classification (Neutral, Joy, Surprise, Sadness, Fear, Anger, Disgust) via rule backend; hooks for LLM/HF

**Interpretation**

* 🧠 Record‑level summaries, affect explanations, simple theme tags (LLM‑friendly hook)

**Visualisation**

* 🗺️ GeoJSON builder + optional **Folium** HTML map
* 📈 Co‑occurrence edge list for graphs (entity co‑mentions)

**CLI & performance**

* Multiprocessing with `--workers` / `--chunksize`
* Optional `--tqdm` progress bars

---

## 📦 Installation

```bash
python -m venv venv
source venv/bin/activate       # on Windows: venv\Scripts\activate
pip install -U pip wheel
pip install -r requirements.txt

# Download at least one spaCy model
python -m spacy download en_core_web_trf   # transformer (higher quality)
# or
python -m spacy download en_core_web_sm    # small (fast)
```

> Packaging tip: expose a console entry point (e.g. `spatio-textual=spatio_textual.cli:main`) so users can call the CLI after `pip install .`.

### 📄 requirements.txt (suggested)

```text
# Base
spacy>=3.6,<4.0
geonamescache>=2.0.0

# Progress bar (optional)
tqdm>=4.66.0

# Visualisation (optional)
folium>=0.15.0

# (Optional) Transformers/LLM backends for richer sentiment/emotion later
# transformers>=4.41.0
# torch>=2.0.0
# openai>=1.0.0
```

### 📂 Resources directory

Place pattern lists under `resources/` (or pass `--resources-dir`). Recognised files:

* `combined_geonouns.txt`, `non_verbals.txt`, `family_terms.txt`, `cleaned_holocaust_camps.txt`

Details of the `spatio_textual/resources/` files:

| File   | Description  |
| ----------------------------- | ---------------------------------------------------------------- |
| `combined_geonouns.txt` | Common geographic feature nouns (e.g., *valley*, *road*, *lake*) |
| `cleaned_holocaust_camps.txt` | Known Holocaust camp names (e.g., *Auschwitz*, *Theresienstadt*) |
| `ambiguous_cities.txt`   | Locations with possible ambiguity (e.g., *Paris*, *Lancaster*)   |
| `non_verbal.txt` | Non-verbal expressions by survivors ([PAUSES], [LAUGHS] etc) in the testimonies |
| `family_terms.txt`  | Family-related entity names (e.g., *mother*, *uncle*)            |

---

## 🚀 Quickstarts

### 🐍 Python (minimal)

```python
from spatio_textual.utils import load_spacy_model, Annotator, save_annotations, load_annotations
from spatio_textual.sentiment import SentimentAnalyzer
from spatio_textual.emotion import EmotionAnalyzer
from spatio_textual.analysis import analyze_records

nlp = load_spacy_model("en_core_web_sm")
ann = Annotator(nlp)

text = "Anne Frank was taken from Amsterdam to Auschwitz."
rec = ann.annotate(text, include_verbs=True)        # entities + verbs
rec["fileId"], rec["segId"], rec["segCount"] = "example", 1, 1
rec["text"] = text

# Affect (rule backends work offline)
sent = SentimentAnalyzer("rule").predict([text])[0]
emo  = EmotionAnalyzer("rule").predict([text])[0]
rec.update({
    "sentiment_label": sent["label"], "sentiment_score": sent["score"],
    "emotion_label": emo["label"],     "emotion_score": emo["score"],
})

# Optional interpretation
[rec] = analyze_records([rec])

save_annotations([rec], "out/sample.jsonl")
df = load_annotations("out/sample.jsonl")
print(df[["fileId","segId","sentiment_label","emotion_label"]].head())
```

### 🧪 Python (files & chunking)

```python
from spatio_textual.annotate import annotate_files

# Chunk each file into `n_segments` sentence-safe segments and annotate
results = annotate_files(["data/"], chunk=True, n_segments=50, include_verbs=True)
```

### 📓 Colab

```bash
# Colab: install runtime deps
pip -q install spacy geonamescache tqdm folium
python -m spacy download en_core_web_trf
```

```python
from spatio_textual.utils import load_spacy_model, Annotator, save_annotations, load_annotations
from spatio_textual.qa import segment_testimony
from spatio_textual.sentiment import SentimentAnalyzer
from spatio_textual.emotion import EmotionAnalyzer
from spatio_textual.analysis import analyze_records

nlp = load_spacy_model("en_core_web_sm")
ann = Annotator(nlp)

text = "Anne Frank was taken from Amsterdam to Auschwitz."
segments = [s.text for s in segment_testimony("Q: " + text, nlp=nlp)]
recs = ann.annotate_texts(segments, file_id="sample", include_text=True)

sent = SentimentAnalyzer("rule")
emo  = EmotionAnalyzer("rule")
spans = [r["text"] for r in recs]
for r, s in zip(recs, sent.predict(spans)):
    r["sentiment_label"], r["sentiment_score"] = s["label"], s["score"]
for r, e in zip(recs, emo.predict(spans)):
    r["emotion_label"], r["emotion_score"] = e["label"], e["score"]

recs = analyze_records(recs)
save_annotations(recs, "sample.jsonl")
df = load_annotations("sample.jsonl")
df.head()
```

### 🖥️ CLI

```bash
# Show environment info
python -m spatio_textual.cli --info --spacy-model en_core_web_sm --resources-dir resources/

# Single file → chunked (~100 segs) → pretty JSON
python -m spatio_textual.cli -i data/sample.txt --pretty

# Whole-file mode (no chunking)
python -m spatio_textual.cli -i data/sample.txt --no-chunk -o out/sample.json --output-format json

# Directory (recursive) → JSONL stream, multiprocessing + tqdm, with verbs
python -m spatio_textual.cli \
  -i corpus/ --glob "*.txt" --workers 6 --chunksize 16 --tqdm --verbs \
  -o out/corpus.jsonl --output-format jsonl

# Testimony segmentation + affect + interpretation
python -m spatio_textual.cli \
  -i data/testimonies/ --glob "*.txt" --tqdm \
  --testimony --sentiment rule --emotion rule --interpret \
  -o out/testimonies.jsonl --output-format jsonl

# List-of-texts mode (JSON array) → TSV
python -m spatio_textual.cli --segments-json segments.json --tqdm -o out/segments.tsv --output-format tsv
```

---

## 🧠 API Reference (essentials)

### utils.py

* `load_spacy_model(model_name='en_core_web_trf', resources_dir=None, add_entity_ruler=True)` → `nlp`
* `split_into_segments(text, n_segments=100, nlp=None)` → `[str]`
* `save_annotations(records, path, fmt=None)` → writes **json/jsonl/tsv/csv**
* `load_annotations(path, fmt=None, ensure_columns=True)` → `pandas.DataFrame`
* `class Annotator(nlp)`:

  * `.annotate(text, include_entities=True, include_verbs=False)`
  * `.annotate_texts(texts, file_id=None, include_text=False, ...)` → `[dict]`
  * `.annotate_file_chunked(path, n_segments=100, ...)` → `[dict]`
  * `.annotate_inputs(inputs, glob_pattern, recursive, ...)` → `[dict]`

### annotate.py (convenience)

* `annotate_text`, `annotate_texts`, `chunk_and_annotate_text`, `chunk_and_annotate_file`, `annotate_files`

### qa.py

* `segment_testimony(text, nlp=None, speaker_patterns=None, join_continuations=True, sentence_safe=True)` → `[Segment]`

  * `Segment`: `text, role, turn_id, is_question, is_answer, qa_pair_id`
* `segment_testimony_file(path, **kwargs)`

### sentiment.py / emotion.py

* `SentimentAnalyzer(backend='rule'|'llm', model_name=None, llm_fn=None).predict(list[str])` → `[{'label','score'}]`
* `EmotionAnalyzer(backend='rule'|'llm', model_name=None, llm_fn=None).predict(list[str])` → `[{'label','score', 'distribution'?}]`

### analysis.py

* `analyze_records(records, llm_fn=None, summarize=True, explain=True, tag_themes=True)` → records + `summary`, `interpretation`, `themes`

### viz.py

* `to_geojson(records, geocoder=None)` → GeoJSON FeatureCollection
* `make_map_geojson(geojson, out_html='map.html')` → saves Folium HTML (if installed)
* `build_cooccurrence(records, nodes=("PERSON","GPE"), window=1)` → `(u,v,w)` edge list

---

## 🧪 Output schema (per record)

Common keys (always present or filled with empty defaults):

```
file, fileId, segId, segCount, entities, verb_data, text?, error?
# Testimony extras
role, turnId, qaPairId, isQuestion, isAnswer
# Affect
sentiment_label, sentiment_score, emotion_label, emotion_score, emotion_dist?
# Interpretation
summary, interpretation, themes
```

`entities` and `verb_data` are arrays (flattened to JSON strings in TSV/CSV).

---

## ⚙️ Performance tips

* Prefer `--workers N` close to your CPU core count; tune `--chunksize` (8–32) for large corpora
* Use `--tqdm` for a progress bar
* For massive analyses, save as JSONL and load with `load_annotations()`

---

## 🧰 Troubleshooting

* **spaCy model not found** → run `python -m spacy download en_core_web_trf` (or `_sm`)
* **Slow transformer model** → try `en_core_web_sm` during development
* **Empty map** → you must supply a geocoder to `to_geojson`
* **Pandas schema mismatches** → `load_annotations(..., ensure_columns=True)` fills missing columns

---

## 📄 License

MIT License. See [LICENSE](./LICENSE).

---

## 🤝 Contributions

Feel free to fork, improve and contribute! Future improvements:

- Better disambiguation of place names
- Map integration and visualisation tools
- CLDW-specific training data integration

---

## 🔗 Acknowledgements

The project is funded in the UK from 2022 to 2025 by ESRC, project reference: ES/W003473/1. We also acknowledge the input and advice from the other members of the project team in generating requirements for our research presented here and the [UCREL Hex](https://www.lancaster.ac.uk/scc/research/research-facilities/hex/) team for providing the compute needs for this project. More details of the project can be found on the website: [Spatial Narratives Project](https://spacetimenarratives.github.io/)

---
