# spatio-textual

<<<<<<< HEAD
**spatio-textual** is a Python library for spatial entity recognition and verb relation extraction from text. It is designed to support spatial analysis in digital humanities projects, with initial applications to:

- the *Corpus of Lake District Writing* (CLDW)
- Holocaust survivors' testimonies (e.g., USC Shoah Foundation archives)

This package leverages spaCy and gazetteer-based classification to identify and label spatial entities such as cities, countries, camps, and geographic nouns, and also extracts action-verb contexts involving these entities.

---

## 🚀 Installation
=======
**1. Clone this repo**
```bash
$ git clone https://github.com/SpaceTimeNarratives/spatio-textual.git
```
**2. Run `setup.sh`**
```bash
$ ./setup.sh
```
This will:
  - Create a venv/ folder
  - Activate it 
  - Install everything from requirements.txt
  - Download the current default model, spacy's `en-core-web-trf` 
>>>>>>> ac5a7335c28214afc1602bae663f2f5f5aac01bf

```bash
<<<<<<< HEAD
pip install spatio-textual
=======
$ source venv/bin/activate
>>>>>>> ac5a7335c28214afc1602bae663f2f5f5aac01bf
```

Or from source:

```bash
<<<<<<< HEAD
git clone https://github.com/your-username/spatio-textual.git
cd spatio-textual
pip install .
=======
$ python entity_annotator.py -i segments.jsonl -o out_dir -r resources -w 4
>>>>>>> ac5a7335c28214afc1602bae663f2f5f5aac01bf
```

---

## 🔍 Example Usage

```python
from spatio_textual import annotate_text

text = "Anne Frank was taken from Amsterdam to Auschwitz."
result = annotate_text(text)
print(result)
```

Output:

```json
{
  "entities": {
    "0": {"text": "Anne Frank", "tag": "PERSON"},
    "25": {"text": "Amsterdam", "tag": "CITY"},
    "39": {"text": "Auschwitz", "tag": "CAMP"}
  },
  "verb_data": [
    {
      "verb": "taken",
      "subject": "Frank",
      "object": "Auschwitz",
      "sentence": "Anne Frank was taken from Amsterdam to Auschwitz."
    }
  ]
}
```

---

## 📁 Resources

The following files support place name classification and disambiguation. They are located in `spatio_textual/resources/`:

| File                          | Description                                                      |
| ----------------------------- | ---------------------------------------------------------------- |
| `combined_geonouns.txt`       | General geographic nouns (e.g., *valley*, *hill*)                |
| `cleaned_holocaust_camps.txt` | Known Holocaust camp names (e.g., *Auschwitz*, *Theresienstadt*) |
| `ambiguous_cities.txt`        | Locations with possible ambiguity (e.g., *Paris*, *York*)        |
| `non_verbal.txt`              | Words often confused with verbs in travel contexts               |
| `family_terms.txt`            | Family-related entity names (e.g., *mother*, *uncle*)            |

You can update or extend these lists to suit your corpus.

---

## 🧩 Components

- `annotate_text(text: str)` → returns a dict with classified spatial entities and verb relations.
- Internally uses:
  - spaCy pipeline (e.g., `en_core_web_trf`)
  - GeonamesCache (cities, countries, etc.)
  - Custom resource lists

---

## 📚 Corpus-Specific Support

### Lake District Writing

- Recognizes landscape terms (e.g., *fells*, *tarns*, *crag*) from `combined_geonouns.txt`
- Can be extended with toponyms of the Lake District

### Holocaust Testimonies

- Annotates camps, movements, and geographic references
- Uses `cleaned_holocaust_camps.txt` and `family_terms.txt`

---

## 🛠 Development

Clone and install with editable mode:

```bash
git clone https://github.com/your-username/spatio-textual.git
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

- spaCy (Explosion AI)
- GeoNamesCache
- Digital Humanities @ Lancaster University

