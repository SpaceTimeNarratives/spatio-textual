from __future__ import annotations

"""
utils.py – spaCy/PlaceNames utilities for single or multi-file annotation.

Adds:
- Optional entity/verb toggles in annotate* methods (entities default True; verbs default False)
- split_into_segments(text, n_segments=100, nlp=None)
- save_annotations(records, path, fmt=None) -> persists as JSON, JSONL, TSV, or CSV
"""

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union, Optional
import csv
import json

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

    files_and_labels = [
        ("combined_geonouns.txt", "GEONOUN"),
        ("non_verbals.txt", "NON-VERBAL"),
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
# Sentence-safe chunker
# ---------------------------------------------------------------------------
def split_into_segments(
    text: str,
    n_segments: int = 100,
    nlp: Optional[spacy.Language] = None,
) -> List[str]:
    """
    Split text into ~n_segments without breaking sentences.
    If fewer sentences than n_segments, returns <= n_segments segments.
    """
    if not text or not text.strip():
        return []

    _nlp = nlp
    if _nlp is None:
        _nlp = spacy.blank("en")
        if "sentencizer" not in _nlp.pipe_names:
            _nlp.add_pipe("sentencizer")

    doc = _nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sents:
        return []

    n = max(1, min(n_segments, len(sents)))
    base, extra = divmod(len(sents), n)
    segments: List[str] = []
    i = 0
    for k in range(n):
        size = base + (1 if k < extra else 0)
        chunk = " ".join(sents[i:i + size])
        if chunk:
            segments.append(chunk)
        i += size
    return segments


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------
def _infer_format(path: Union[str, Path], fmt: Optional[str]) -> str:
    if fmt:
        return fmt.lower()
    ext = str(path).lower()
    if ext.endswith(".jsonl") or ext.endswith(".ndjson"):
        return "jsonl"
    if ext.endswith(".json"):
        return "json"
    if ext.endswith(".tsv"):
        return "tsv"
    if ext.endswith(".csv"):
        return "csv"
    # default to json
    return "json"


def _flatten_for_table(rec: dict) -> dict:
    """Flatten nested lists into JSON strings for TSV/CSV friendliness."""
    out = dict(rec)
    if "entities" in out and not isinstance(out["entities"], (str, type(None))):
        out["entities"] = json.dumps(out["entities"], ensure_ascii=False)
    if "verb_data" in out and not isinstance(out["verb_data"], (str, type(None))):
        out["verb_data"] = json.dumps(out["verb_data"], ensure_ascii=False)
    return out


def save_annotations(records: List[dict], path: Union[str, Path], fmt: Optional[str] = None) -> None:
    """
    Persist annotations in pandas-friendly formats: json, jsonl, tsv, csv.
    - json: writes a single JSON array
    - jsonl/ndjson: one JSON object per line
    - tsv/csv: flattens entities/verb_data as JSON strings
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    kind = _infer_format(p, fmt)

    if kind == "json":
        p.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if kind == "jsonl":
        with p.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return

    # tabular
    flat = [_flatten_for_table(r) for r in records]
    fieldnames = sorted({k for r in flat for k in r.keys()})
    delimiter = "\t" if kind == "tsv" else ","
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for row in flat:
            w.writerow({k: row.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# PlaceNames helper for fine-grained place classification
# ---------------------------------------------------------------------------
class PlaceNames:
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

    def annotate(self, text: str, *, include_entities: bool = True, include_verbs: bool = False) -> dict:
        """
        Annotate a single text string.
        - include_entities: emit 'entities' (default True)
        - include_verbs: emit 'verb_data' (default False)
        """
        doc = self.nlp(text)
        result = {}

        entities = []
        if include_entities:
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
        result["entities"] = entities

        verb_data = self.extract_verbs(doc) if include_verbs else []
        result["verb_data"] = verb_data

        return result

    # ------------------ File helpers ------------------

    def annotate_file(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
        errors: str = "ignore",
        include_text: bool = False,
        include_entities: bool = True,
        include_verbs: bool = False,
    ) -> Dict:
        p = Path(path)
        text = p.read_text(encoding=encoding, errors=errors)
        result = self.annotate(text, include_entities=include_entities, include_verbs=include_verbs)
        result["file"] = str(p)
        result["fileId"] = p.stem
        result["segId"] = 1
        result["segCount"] = 1
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
        include_entities: bool = True,
        include_verbs: bool = False,
    ) -> List[Dict]:
        files = list(self._resolve_input_files(inputs, glob_pattern, recursive))
        results: List[Dict] = []
        for f in files:
            try:
                results.append(
                    self.annotate_file(
                        f, encoding=encoding, errors=errors,
                        include_text=include_text,
                        include_entities=include_entities, include_verbs=include_verbs
                    )
                )
            except Exception as e:
                results.append({"file": str(f), "error": repr(e)})
        return results

    def annotate_texts(
        self,
        texts: Sequence[str],
        file_id: str | None = None,
        start_seg_id: int = 1,
        include_text: bool = False,
        include_entities: bool = True,
        include_verbs: bool = False,
    ) -> List[Dict]:
        out: List[Dict] = []
        seg_id = start_seg_id
        for t in texts:
            res = self.annotate(t, include_entities=include_entities, include_verbs=include_verbs)
            if file_id is not None:
                res["fileId"] = file_id
            res["segId"] = seg_id
            if include_text:
                res["text"] = t
            out.append(res)
            seg_id += 1
        if file_id is not None:
            for r in out:
                r["segCount"] = len(out)
        return out

    def annotate_file_chunked(
        self,
        path: str | Path,
        n_segments: int = 100,
        encoding: str = "utf-8",
        errors: str = "ignore",
        include_text: bool = False,
        include_entities: bool = True,
        include_verbs: bool = False,
    ) -> List[Dict]:
        p = Path(path)
        raw = p.read_text(encoding=encoding, errors=errors)
        segments = split_into_segments(raw, n_segments=n_segments, nlp=self.nlp)
        file_id = p.stem
        total = len(segments)
        results: List[Dict] = []
        for idx, seg in enumerate(segments, 1):
            res = self.annotate(seg, include_entities=include_entities, include_verbs=include_verbs)
            res.update({
                "file": str(p),
                "fileId": file_id,
                "segId": idx,
                "segCount": total,
            })
            if include_text:
                res["text"] = seg
            results.append(res)
        return results

    @staticmethod
    def _resolve_input_files(
        inputs: Union[str, Path, Sequence[Union[str, Path]]],
        glob_pattern: str = "*.txt",
        recursive: bool = True,
    ) -> Iterable[Path]:
        if isinstance(inputs, (str, Path)):
            p = Path(inputs)
            if p.exists():
                if p.is_file():
                    yield p
                elif p.is_dir():
                    yield from (p.rglob(glob_pattern) if recursive else p.glob(glob_pattern))
            else:
                yield from Path().glob(str(inputs))
        else:
            for item in inputs:
                yield from Annotator._resolve_input_files(item, glob_pattern, recursive)


__all__ = [
    "resources_dir",
    "load_spacy_model",
    "split_into_segments",
    "save_annotations",
    "PlaceNames",
    "Annotator",
]
