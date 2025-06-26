# spatio-textual

**spatio-textual** is a Python library for spatial entity recognition and verb relation extraction from text. It created as part of the [Spatial Narratives Project](https://spacetimenarratives.github.io/) and is designed to support spatio-textual annotation, analysis and visualization in digital humanities projects, with initial applications to:

- the *Corpus of Lake District Writing* (CLDW)
- Holocaust survivors' testimonies (e.g., USC Shoah Foundation archives)

This package leverages spaCy and gazetteer-based classification to identify and label spatial entities such as cities, countries, camps, and geographic nouns, and also extracts action-verb contexts involving these entities.

---

## 🚀 Installation

```bash
$ source venv/bin/activate
```

Or from source:

```bash
git clone https://github.com/SpaceTimeNarratives/spatio-textual.git
cd spatio-textual
pip install .
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

```py
{
  'entities': [
    {'start_char': 0, 'token': 'Anne Frank', 'tag': 'PERSON'},
    {'start_char': 26, 'token': 'Amsterdam', 'tag': 'CITY'},
    {'start_char': 39, 'token': 'Auschwitz', 'tag': 'CAMP'}
    ],
  'verb_data': [
    {'sent-id': 0, 'verb': 'taken', 'subject': 'Anne Frank', 'object': 'Amsterdam',
   'sentence': 'Anne Frank was taken from Amsterdam to Auschwitz.'}]}
```

---

## 📁 Resources

The following files support place name classification and disambiguation. They are located in `spatio_textual/resources/`:

| File                          | Description                                                      |
| ----------------------------- | ---------------------------------------------------------------- |
| `combined_geonouns.txt`       | Common geographic feature nouns (e.g., *valley*, *road*, *lake*) |
| `cleaned_holocaust_camps.txt` | Known Holocaust camp names (e.g., *Auschwitz*, *Theresienstadt*) |
| `ambiguous_cities.txt`        | Locations with possible ambiguity (e.g., *Paris*, *Lancaster*)   |
| `non_verbal.txt`              | Non-verbal expressions by survivors ([PAUSES], [LAUGHS] etc) in the testimonies |
| `family_terms.txt`            | Family-related entity names (e.g., *mother*, *uncle*)            |

You can update or extend these lists to suit your corpus or task.

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

- spaCy (Explosion AI)
- [Spatial Narratives Project](https://spacetimenarratives.github.io/)
