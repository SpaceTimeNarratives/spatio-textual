# spatio-textual ✨

A Python library for spatial textual analysis created as part of the [Spatial Narratives Project](https://spacetimenarratives.github.io/). It supports spatio-textual annotation, analysis and visualization for digital humanities projects, with initial applications to:

- **Corpus of Lake District Writing (CLDW)**
- **Holocaust survivors' testimonies** (e.g., USC Shoah Foundation archives)

This realease adds **sentence-safe chunking**, **list-of-texts input**, **file/segment IDs in output**, **JSON/JSONL saving**, and a **multiprocessing CLI**.

---

## 🧭 Contents

- `utils.py` — core utilities (spaCy loader, `PlaceNames`, `Annotator`, `split_into_segments`)
- `annotate.py` — simple Python API entry points
- `cli.py` — command-line interface for single files, directories, or in-memory segment lists
- `resources/` — optional term lists for the `EntityRuler` and classification
  - `combined_geonouns.txt`
  - `non_verbals.txt`
  - `family_terms.txt`
  - `cleaned_holocaust_camps.txt`

Details of the file contents `spatio_textual/resources/`

| File   | Description  |
| ----------------------------- | ---------------------------------------------------------------- |
| `combined_geonouns.txt` | Common geographic feature nouns (e.g., *valley*, *road*, *lake*) |
| `cleaned_holocaust_camps.txt` | Known Holocaust camp names (e.g., *Auschwitz*, *Theresienstadt*) |
| `ambiguous_cities.txt`   | Locations with possible ambiguity (e.g., *Paris*, *Lancaster*)   |
| `non_verbal.txt` | Non-verbal expressions by survivors ([PAUSES], [LAUGHS] etc) in the testimonies |
| `family_terms.txt`  | Family-related entity names (e.g., *mother*, *uncle*)            |

---

## 📦 Installation

### 1. Clone
```bash
git clone https://github.com/SpaceTimeNarratives/spatio-textual.git
cd spatio-textual
```

### 2. Environment
Use the provided script **or** do it manually.

**Script (if `setup.sh` exists):**
```bash
./setup.sh
```

**Manual setup:**
```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -U pip wheel
pip install -r requirements.txt

# spaCy model (choose one)
python -m spacy download en_core_web_trf   # transformer (recommended, larger)
# OR
python -m spacy download en_core_web_sm    # small (faster, lower quality)
```

### 3. (Optional) Resources
By default, the CLI and APIs look for `resources/` next to the code.  
To use a custom resources folder, pass `--resources-dir /path/to/resources` (CLI) or `resources_dir=...` in Python.

> 💡 **Pip-installable package**: When packaging, add a `console_scripts` entry point (e.g., `spatio-textual=cli:main`) so `spatio-textual` is available on the PATH after `pip install .`.

---

## 🚀 Quick Start (CLI)

The CLI supports:
- **Sentence-safe chunking** (`--segments N`, default `100`) – no broken sentences across segments.
- **Whole-file mode** (`--no-chunk`) – process the entire file as one record.
- **List-of-texts input** from a JSON array (`--segments-json`).
- **Multiprocessing** (`--workers`, `--chunksize`).
- **EntityRuler toggle** (`--no-entity-ruler`).
- **JSON** (array) or **JSONL** (one object per line) output.

> Tip: Run `python cli.py --info` to verify the model, pipes, and resources.

### Show environment info
```bash
python cli.py --info --spacy-model en_core_web_trf --resources-dir resources/
```

### Single file → chunked (~100 segments) → pretty JSON to stdout
```bash
echo "Anne Frank was taken from Amsterdam to Auschwitz." > data/sample.txt
python cli.py -i data/sample.txt --pretty
```

### Single file → chunked (~50 segments) → save JSON array
```bash
python cli.py -i data/sample.txt --segments 50 -o out/sample.json --output-format json
```

### Whole-file mode (no chunking) → JSON array (single object)
```bash
python cli.py -i data/sample.txt --no-chunk -o out/sample_whole.json --output-format json
```

### Directory (recursive) → JSONL (stream) with multiprocessing
```bash
python cli.py -i corpus/ --glob "*.txt" --workers 6 --chunksize 16 \
  -o out/corpus.jsonl --output-format jsonl --progress
```

### Mixed inputs (files + dir + glob) → JSON (array)
```bash
python cli.py -i data/a.txt data/ "corpus/**/*.md" \
  -o out/mixed.json --output-format json
```

### In-memory list of segments/texts from JSON
`segments.json` can be either strings, or objects with `"text"` and optional `"fileId"` / `"segId"`:
```json
[
  {"text": "Anne Frank was taken from Amsterdam to Auschwitz.", "fileId": "sample", "segId": 1},
  "Anne Frank was taken from Amsterdam to Auschwitz."
]
```
Run:
```bash
python cli.py --segments-json segments.json -o out/segments.json --output-format json
```

### Dry run (see which files would be processed)
```bash
python cli.py -i "corpus/**/*.txt" --dry-run
```

### Disable EntityRuler (speed tests)
```bash
python cli.py -i data/sample.txt --no-chunk --no-entity-ruler --pretty
```
---

## 🧪 Output schema

### Chunked records (default)
```jsonc
{
  "file": "path/to/sample.txt",   // present for file-based runs
  "fileId": "sample",             // filename without extension
  "segId": 1,                     // 1-based segment index
  "segCount": 1,                  // total number of segments in the file
  "entities": [ { "start_char": 0, "token": "Anne Frank", "tag": "PERSON" }, { "token": "Amsterdam", "tag": "CITY" }, { "token": "Auschwitz", "tag": "CAMP" } ],
  "verb_data": [ { "sent-id": 0, "verb": "taken", "subject": "Anne Frank", "object": "Auschwitz", "sentence": "Anne Frank was taken from Amsterdam to Auschwitz." } ],
  "text": "..."                   // included only if --include-text
}
```

### Whole-file records (`--no-chunk`)
```jsonc
{
  "file": "path/to/sample.txt",
  "fileId": "sample",
  "segId": 1,
  "segCount": 1,
  "entities": [...],
  "verb_data": [...],
  "text": "..."   // if --include-text
}
```

### In-memory list mode (`--segments-json`)
- If `fileId` is provided in the input, it is preserved and used to reset `segId` per file.
- If `segId` is provided in the input, it overrides the auto numbering.
- Otherwise, `segId` starts at 1 in each implicit group.

---

## 🧠 Python API

### High-level (simple) — `annotate.py`

```python
from spatio_textual.annotate import (
    annotate_text,
    annotate_texts,
    chunk_and_annotate_text,
    chunk_and_annotate_file,
)

# Single string
text = "Anne Frank was taken from Amsterdam to Auschwitz."
res1 = annotate_text(text)

# List of strings (segments) with optional fileId and segId numbering
chunks = [text, text]
res2 = annotate_texts(chunks, file_id="sample", include_text=False)

# Chunk a long text into ~80 sentence-safe segments and annotate
long_text = text * 50
res3 = chunk_and_annotate_text(long_text, n_segments=80, file_id="sample")

# Chunk a file and annotate segments
res4 = chunk_and_annotate_file("data/sample.txt", n_segments=100, include_text=False)
```

### Lower-level / flexible — `utils.py`

```python
from spatio_textual.utils import load_spacy_model, Annotator, split_into_segments

# Choose a model
nlp = load_spacy_model("en_core_web_trf", resources_dir="resources")  # or en_core_web_sm

# Build annotator
ann = Annotator(nlp, resources_dir="resources")

# Sentence-safe chunking helper
text = "Anne Frank was taken from Amsterdam to Auschwitz."
segments = split_into_segments(text * 30, n_segments=10, nlp=nlp)

# Annotate a list of texts
out = ann.annotate_texts(segments, file_id="sample", start_seg_id=1, include_text=False)

# Annotate a single file (whole-file mode)
single = ann.annotate_file("data/sample.txt")

# Annotate a file with chunking
chunked = ann.annotate_file_chunked("data/sample.txt", n_segments=5, include_text=True)
```

---

## 📓 Colab / Notebook Usage

```python
# Install (library + optional transformers extras)
!pip -q install spacy geonamescache
# Optional for transformer model:
# !pip -q install spacy-transformers torch

# Download a model
!python -m spacy download en_core_web_sm  # or en_core_web_trf

# Use the library
from spatio_textual.annotate import annotate_text, chunk_and_annotate_text

text = "Anne Frank was taken from Amsterdam to Auschwitz."
print(annotate_text(text))

# Chunk and annotate a longer text
long_text = text * 30
res = chunk_and_annotate_text(long_text, n_segments=10, file_id="sample")
len(res), res[0]
```

---

## 📄 requirements.txt

Add these to your `requirements.txt` for packaging and installation:

```text
spacy>=3.6,<4.0
geonamescache>=2.0.0

# Optional for transformer model (en_core_web_trf):
spacy-transformers>=1.3.4
# PyTorch is required by spacy-transformers; install a build matching your platform/compute
# Example (CPU only):
# torch>=2.0.0
```

> If you plan to ship wheels that pre-pin models, you can also add model packages like `en_core_web_sm` as an extra install step or document them under **Installation**.

Add these to your `requirements.txt` for packaging and installation:

```text
spacy>=3.6,<4.0
geonamescache>=2.0.0

# Optional for transformer model (en_core_web_trf):
spacy-transformers>=1.3.4
# PyTorch is required by spacy-transformers; install a build matching your platform/compute
# Example (CPU only):
# torch>=2.0.0
```

> If you plan to ship wheels that pre-pin models, you can also add model packages like `en_core_web_sm` as an extra install step or document them under **Installation**.

---

## 📝 Notes & Tips

* **Resources**: Place custom term lists under `resources/` or point `--resources-dir` to your folder.
* **EntityRuler**: Enabled by default. Use `--no-entity-ruler` to disable.
* **Multiprocessing**: Use `--workers` close to the number of CPU cores and a `--chunksize` of 8–32 for large corpora.
* **Windows/macOS**: The CLI uses the `spawn` start method, which is compatible across platforms.
* **Models**: `en_core_web_trf` (higher quality, slower) vs `en_core_web_sm` (faster, smaller).
* **Reproducibility**: Pin `spacy`, `geonamescache`, and model versions in `requirements.txt`.

---

## 🔄 Previous command replaced

The earlier README referenced:

```bash
python entity_annotator.py -i segments.jsonl -o out_dir -r resources -w 4
```

That workflow is now covered by the new unified CLI. For example:

```bash
# Approximately equivalent: process a list of segments from a JSON file
python cli.py --segments-json segments.json -o out_dir/segments.json --output-format json --workers 4
```
---

## 📚 Corpus-Specific Support

### Lake District Writing

- Recognizes landscape terms (e.g., *valley*, *road*, *lake*) from `combined_geonouns.txt`
- Can be extended with toponyms of the Lake District

### Holocaust Testimonies

- Annotates camps, movements, and geographic references
- Uses `cleaned_holocaust_camps.txt` and `family_terms.txt`

---

## 🛠 Development

Clone and install with editable mode:

```bash
git clone https://github.com/SpaceTimeNarratives/spatio-textual.git
cd spatio-textual
pip install -e .
```

Run tests (coming soon):

```bash
pytest tests/
```

---

## 📄 License

GNU GENERAL PUBLIC LICENSE. See [LICENSE](./LICENSE).

---

## 🤝 Contributions

Feel free to fork, improve and contribute! Future improvements:

- Better disambiguation of place names
- Map integration and visualisation tools
- CLDW-specific training data integration

---

## 🔗 Acknowledgements

The project is funded in the UK from 2022 to 2025 by ESRC, project reference: ES/W003473/1. We also acknowledge the input and advice from the other members of the project team in generating requirements for our research presented here and the [UCREL Hex](https://www.lancaster.ac.uk/scc/research/research-facilities/hex/) team for providing the compute needs for this project. More details of the project can be found on the website: [Spatial Narratives Project](https://spacetimenarratives.github.io/)
