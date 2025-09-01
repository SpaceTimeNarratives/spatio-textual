from __future__ import annotations

"""
utils.py – spaCy/PlaceNames utilities for single or multi-file annotation.

Key features:
- load_spacy_model(model_name, resources_dir=None, add_entity_ruler=True)
  Loads a spaCy pipeline and (optionally) attaches an EntityRuler whose patterns
  are read from `resources_dir` (falls back to this module's resources folder).

- PlaceNames(resources_dir=None)
  Loads GeoNames data and curated lists from `resources_dir` to refine place tags.

- Annotator(nlp, resources_dir=None)
  Wraps a spaCy pipeline to:
    - annotate(text) -> dict
    - extract_verbs(doc) -> list
    - annotate_file(path) -> dict
    - annotate_inputs(inputs, ...) -> list[dict]
  Includes a robust file resolver for single files, directories, glob patterns,
  and sequences of any of the above.

Multiprocessing:
- Safe to construct one spaCy pipeline + Annotator PER PROCESS.
- Do not share spaCy Language objects across processes.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

import spacy
from geonamescache import GeonamesCache


# ---------------------------------------------------------------------------
# Backward-compatible module-level resources path
# ---------------------------------------------------------------------------
_MODULE_RESOURCES_DIR = Path(__file__).parent / "resources"
# Keep this name for backwards compatibility with older imports
resources_dir = _MODULE_RESOURCES_DIR


# ---------------------------------------------------------------------------
# spaCy loader with optional EntityRuler wired to resources_dir
# ---------------------------------------------------------------------------
def load_spacy_model(
    model_name: str = "en_core_web_trf",
    resources_dir: Union[str, Path, None] = None,
    add_entity_ruler: bool = True,
) -> spacy.Language:
    """
    Load a spaCy pipeline and (optionally) attach an EntityRuler whose patterns
    are read from `resources_dir`. Missing pattern files are skipped gracefully.

    Args:
        model_name: spaCy model to load (e.g., 'en_core_web_trf', 'en_core_web_sm').
        resources_dir: Optional folder containing pattern files. Defaults to the
                       package resources folder if not provided.
        add_entity_ruler: If False, no EntityRuler is added.

    Returns:
        A configured spaCy Language pipeline.
    """
    nlp = spacy.load(model_name)

    # Merge multi-token entities so they behave as single tokens downstream.
    if "merge_entities" not in nlp.pipe_names:
        nlp.add_pipe("merge_entities")

    if not add_entity_ruler:
        return nlp

    base = Path(resources_dir) if resources_dir else _MODULE_RESOURCES_DIR

    # Ensure the EntityRuler is before NER for maximal effect.
    ruler = (
        nlp.get_pipe("entity_ruler")
        if "entity_ruler" in nlp.pipe_names
        else nlp.add_pipe("entity_ruler", before="ner")
    )

    def _load_patterns_safe(label: str, filename: str) -> List[Dict]:
        path = base / filename
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as f:
            return [{"label": label, "pattern": line.strip()} for line in f if line.strip()]

    # Keep filenames consistent with your resources folder.
    files_and_labels = [
        ("combined_geonouns.txt", "GEONOUN"),
        ("non_verbals.txt", "NON-VERBAL"),          # unified plural form
        ("family_terms.txt", "FAMILY"),
        ("cleaned_holocaust_camps.txt", "CAMP"),
    ]

    patterns: List[Dict] = []
    for filename, label in files_and_labels:
        patterns.extend(_load_patterns_safe(label, filename))

    if patterns:
        ruler.add_patterns(patterns)

    return nlp


# ---------------------------------------------------------------------------
# PlaceNames helper for fine-grained place classification
# ---------------------------------------------------------------------------
class PlaceNames:
    """
    Classify FAC/GPE/LOC/ORG into finer classes using GeoNames + curated lists.
    Construct per process in multiprocessing contexts.
    """

    def __init__(self, resources_dir: Union[str, Path, None] = None):
        self.resources_dir = Path(resources_dir) if resources_dir else _MODULE_RESOURCES_DIR
        self.gc = GeonamesCache()
        self._load_geo_data()
        self._load_lists()

    def _load_geo_data(self) -> None:
        cities = self.gc.get_cities()
        states = self.gc.get_us_states()
        countries = self.gc.get_countries()
        continents = self.gc.get_continents()

        self.city_names = sorted({c["name"] for c in cities.values()} | {"New York"}, key=len, reverse=True)
        self.state_names = sorted({s["name"] for s in states.values()}, key=len, reverse=True)
        self.country_names = sorted(
            {c["name"] for c in countries.values()} | {"America", "the United States", "Czechoslovakia"},
            key=len,
            reverse=True,
        )
        self.continent_names = sorted({c["name"] for c in continents.values()}, key=len, reverse=True)

    def _load_lists(self) -> None:
        base = self.resources_dir

        def read_list(fname: str) -> List[str]:
            path = base / fname
            if not path.exists():
                return []
            return sorted(
                {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()},
                key=len,
                reverse=True,
            )

        self.geonouns = read_list("combined_geonouns.txt")
        self.camps = read_list("cleaned_holocaust_camps.txt")
        self.ambiguous_cities = read_list("ambiguous_cities.txt")
        # Attribute name kept as 'non_verbal' for backward compatibility; file is plural 'non_verbals.txt'
        self.non_verbal = read_list("non_verbals.txt")
        self.family = read_list("family_terms.txt")

    def classify(self, text: str, label: str) -> str:
        if label in {"FAC", "GPE", "LOC", "ORG"}:
            if text in self.continent_names:
                return "CONTINENT"
            if text in self.country_names:
                return "COUNTRY"
            if text in self.state_names:
                return "US-STATE"
            if text in self.city_names:
                return "CITY"
            if text in self.camps:
                return "CAMP"
            return "PLACE"
        return label


# ---------------------------------------------------------------------------
# Annotator: core annotation + I/O helpers
# ---------------------------------------------------------------------------
class Annotator(PlaceNames):
    """
    Wraps a spaCy pipeline and provides:
      - annotate(text) -> dict
      - extract_verbs(doc) -> list
      - annotate_file(path, ...) -> dict
      - annotate_inputs(inputs, ...) -> list[dict]

    Notes:
      - Instantiate per process for multiprocessing safety.
      - `resources_dir` controls both EntityRuler patterns (via load_spacy_model)
        and curated lists here.
    """

    def __init__(self, nlp: spacy.Language, resources_dir: Union[str, Path, None] = None):
        super().__init__(resources_dir)
        self.nlp = nlp

    def extract_verbs(self, doc: spacy.tokens.Doc) -> list:
        data = []
        for sentid, sent in enumerate(doc.sents):
            for token in sent:
                if token.pos_ == "VERB":
                    subj = [c.text for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
                    obj = [c.text for c in token.children if c.dep_ == "dobj"] + [
                        gc.text
                        for prep in token.children
                        if prep.dep_ == "prep"
                        for gc in prep.children
                        if gc.dep_ == "pobj"
                    ]
                    data.append(
                        {
                            "sent-id": sentid,
                            "verb": token.text,
                            "subject": subj[0] if subj else "",
                            "object": obj[0] if obj else "",
                            "sentence": sent.text,
                        }
                    )
        return data

    def annotate(self, text: str) -> dict:
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in {
                "PERSON",
                "FAC",
                "GPE",
                "LOC",
                "ORG",
                "DATE",
                "TIME",
                "EVENT",
                "QUANTITY",
                "GEONOUN",
                "NON-VERBAL",
                "FAMILY",
                "CAMP",
            }:
                tag = self.classify(ent.text, ent.label_) if ent.label_ in {"FAC", "GPE", "LOC", "ORG"} else ent.label_
                entities.append({"start_char": ent.start_char, "token": ent.text, "tag": tag})

        verb_data = self.extract_verbs(doc)
        return {"entities": entities, "verb_data": verb_data}

    # ------------------ File helpers ------------------

    def annotate_file(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
        errors: str = "ignore",
        include_text: bool = False,
    ) -> Dict:
        """
        Annotate a single text file.

        Returns:
            dict with keys: 'file', 'entities', 'verb_data' and optionally 'text'
        """
        p = Path(path)
        text = p.read_text(encoding=encoding, errors=errors)
        result = self.annotate(text)
        result["file"] = str(p)
        if include_text:
            result["text"] = text
        return result

    def annotate_inputs(
        self,
        inputs: Union[str, Path, Sequence[Union[str, Path]]],
        glob_pattern: str = "*.txt",
        recursive: bool = True,
        encoding: str = "utf-8",
        errors: str = "ignore",
        include_text: bool = False,
    ) -> List[Dict]:
        """
        Annotate one or many inputs:
          - a single file path
          - a directory (scans for files matching glob_pattern)
          - a glob pattern string (e.g., 'data/**/*.txt')
          - a list/sequence of any of the above

        Returns:
            list of per-file annotation dicts
        """
        files = list(self._resolve_input_files(inputs, glob_pattern, recursive))
        results: List[Dict] = []
        for f in files:
            try:
                results.append(
                    self.annotate_file(f, encoding=encoding, errors=errors, include_text=include_text)
                )
            except Exception as e:
                results.append({"file": str(f), "error": repr(e)})
        return results

    @staticmethod
    def _resolve_input_files(
        inputs: Union[str, Path, Sequence[Union[str, Path]]],
        glob_pattern: str = "*.txt",
        recursive: bool = True,
    ) -> Iterable[Path]:
        """
        Yield Path objects for all files represented by 'inputs'.
          - If inputs is a file -> yield it.
          - If inputs is a dir -> yield files matching glob_pattern (recursive or not).
          - If inputs looks like a glob pattern -> treat as glob pattern.
          - If inputs is a sequence -> resolve each element recursively.
        """
        if isinstance(inputs, (str, Path)):
            p = Path(inputs)
            if p.exists():
                if p.is_file():
                    yield p
                elif p.is_dir():
                    yield from (p.rglob(glob_pattern) if recursive else p.glob(glob_pattern))
            else:
                # Treat as a glob pattern relative to CWD (e.g., "data/**/*.txt")
                yield from Path().glob(str(inputs))
        else:
            for item in inputs:
                yield from Annotator._resolve_input_files(item, glob_pattern, recursive)


__all__ = [
    "resources_dir",
    "load_spacy_model",
    "PlaceNames",
    "Annotator",
]