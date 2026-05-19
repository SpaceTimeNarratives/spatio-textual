from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import spacy
from spacy.language import Language
from spacy.tokens import Doc

DEFAULT_RESOURCES_DIR = Path(__file__).parent / "resources"
STANDARD_COLUMNS = [
    "file", "fileId", "segId", "segCount", "entities", "verb_data", "text", "error",
    "role", "turnId", "qaPairId", "isQuestion", "isAnswer",
    "sentiment_label", "sentiment_score", "emotion_label", "emotion_score", "emotion_dist",
    "summary", "interpretation", "themes",
]

PLACE_LABELS = {"GPE", "LOC", "FAC", "COUNTRY", "CITY", "CONTINENT", "CAMP", "REGION", "PLACE", "GEONOUN"}
RESOURCE_LABELS = {
    "combined_geonouns.txt": "GEONOUN",
    "non_verbals.txt": "NON-VERBAL",
    "non_verbal.txt": "NON-VERBAL",
    "ht_non_verbals.txt": "NON-VERBAL",
    "family_terms.txt": "FAMILY",
    "cleaned_holocaust_camps.txt": "CAMP",
    "ambiguous_cities.txt": "CITY",
}

COUNTRY_ALIASES = {"america", "united states", "the united states", "usa", "u.s.", "u.s.a.", "england", "scotland", "wales"}
CONTINENTS = {"africa", "asia", "europe", "north america", "south america", "oceania", "antarctica"}


def _read_terms(path: Path) -> list[str]:
    if not path.exists():
        return []
    terms: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        terms.append(item)
    return terms


@lru_cache(maxsize=8)
def _cached_model(model_name: str, resources_key: str, add_entity_ruler: bool, disable_key: str) -> Language:
    disable = tuple(x for x in disable_key.split(",") if x)
    try:
        nlp = spacy.load(model_name, disable=list(disable))
    except Exception:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    if add_entity_ruler:
        _add_entity_ruler(nlp, Path(resources_key))
    return nlp


def load_spacy_model(
    model_name: str = "en_core_web_sm",
    resources_dir: Union[str, Path, None] = None,
    add_entity_ruler: bool = True,
    disable: Optional[Sequence[str]] = None,
) -> Language:
    """Load spaCy once per process and add optional EntityRuler patterns.

    The loader falls back to a blank English pipeline with a sentencizer when the
    requested model is not installed. This keeps demos, HF Spaces and tests usable
    without a large model download.
    """
    base = Path(resources_dir) if resources_dir else DEFAULT_RESOURCES_DIR
    return _cached_model(str(model_name), str(base.resolve()), bool(add_entity_ruler), ",".join(disable or []))


def _add_entity_ruler(nlp: Language, resources_dir: Path) -> None:
    if "entity_ruler" in nlp.pipe_names:
        ruler = nlp.get_pipe("entity_ruler")
    else:
        before = "ner" if "ner" in nlp.pipe_names else None
        ruler = nlp.add_pipe("entity_ruler", before=before, config={"overwrite_ents": False})
    existing = set()
    try:
        existing = {(p.get("label"), str(p.get("pattern"))) for p in ruler.patterns}
    except Exception:
        existing = set()
    patterns: list[dict[str, str]] = []
    for filename, label in RESOURCE_LABELS.items():
        for term in _read_terms(resources_dir / filename):
            key = (label, term)
            if key not in existing:
                patterns.append({"label": label, "pattern": term})
    if patterns:
        ruler.add_patterns(patterns)


def _sentencizer() -> Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def split_into_segments(
    text: str,
    n_segments: Optional[int] = None,
    nlp: Optional[Language] = None,
    max_chars: Optional[int] = 14000,
    overlap_chars: int = 0,
) -> list[str]:
    """Sentence-safe segmentation for long narratives.

    Use ``n_segments`` for normalised narrative progression or ``max_chars`` for
    API-friendly chunks. Overlap is backed up to sentence boundaries.
    """
    if not text or not text.strip():
        return []
    _nlp = nlp or _sentencizer()
    doc = _nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sents:
        return [text.strip()]

    if n_segments and n_segments > 0:
        n = max(1, min(int(n_segments), len(sents)))
        base, extra = divmod(len(sents), n)
        out: list[str] = []
        pos = 0
        for i in range(n):
            size = base + (1 if i < extra else 0)
            chunk = " ".join(sents[pos:pos + size]).strip()
            if chunk:
                out.append(chunk)
            pos += size
        return out

    if not max_chars or max_chars <= 0:
        return [text.strip()]

    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for sent in sents:
        sent_len = len(sent) + (1 if cur else 0)
        if cur and cur_len + sent_len > max_chars:
            chunks.append(cur)
            if overlap_chars > 0:
                overlap: list[str] = []
                olen = 0
                for old in reversed(cur):
                    if olen + len(old) > overlap_chars and overlap:
                        break
                    overlap.insert(0, old)
                    olen += len(old) + 1
                cur = overlap[:]
                cur_len = sum(len(x) + 1 for x in cur)
            else:
                cur = []
                cur_len = 0
        cur.append(sent)
        cur_len += sent_len
    if cur:
        chunks.append(cur)
    return [" ".join(c).strip() for c in chunks if c]


def classify_place(text: str, label: str) -> str:
    low = text.strip().lower()
    if label == "CAMP":
        return "CAMP"
    if label == "GEONOUN":
        return "GEONOUN"
    if low in CONTINENTS:
        return "CONTINENT"
    if low in COUNTRY_ALIASES:
        return "COUNTRY"
    if label in {"GPE", "LOC", "FAC"}:
        return "PLACE"
    return label or "UNKNOWN"


def _token_offsets(doc: Doc, start_char: int, end_char: int) -> tuple[Optional[int], Optional[int]]:
    start_tok = end_tok = None
    for i, tok in enumerate(doc):
        if start_tok is None and tok.idx <= start_char < tok.idx + len(tok):
            start_tok = i
        if tok.idx < end_char <= tok.idx + len(tok):
            end_tok = i + 1
            break
    return start_tok, end_tok


@dataclass
class EntityRecord:
    text: str
    label: str
    start_char: int
    end_char: int
    start_token: Optional[int] = None
    end_token: Optional[int] = None
    place_type: Optional[str] = None
    confidence: Optional[float] = None
    source: str = "spacy"


class Annotator:
    """High-level entity and verb annotator with file and segment helpers."""

    def __init__(self, nlp: Optional[Language] = None, resources_dir: Union[str, Path, None] = None):
        self.nlp = nlp or load_spacy_model(resources_dir=resources_dir)
        self.resources_dir = Path(resources_dir) if resources_dir else DEFAULT_RESOURCES_DIR

    def annotate(
        self,
        text: str,
        include_entities: bool = True,
        include_verbs: bool = False,
        include_text: bool = False,
    ) -> dict[str, Any]:
        rec: dict[str, Any] = {"entities": [], "verb_data": [], "error": None}
        if include_text:
            rec["text"] = text
        try:
            doc = self.nlp(text or "")
            if include_entities:
                rec["entities"] = self._entities(doc)
            if include_verbs:
                rec["verb_data"] = self._verbs(doc)
        except Exception as exc:
            rec["error"] = str(exc)
        return rec

    def _entities(self, doc: Doc) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ent in doc.ents:
            start_tok, end_tok = _token_offsets(doc, ent.start_char, ent.end_char)
            label = ent.label_
            place_type = classify_place(ent.text, label) if label in PLACE_LABELS else None
            out.append(asdict(EntityRecord(
                text=ent.text,
                label=label,
                start_char=ent.start_char,
                end_char=ent.end_char,
                start_token=start_tok,
                end_token=end_tok,
                place_type=place_type,
            )))
        return out

    def _verbs(self, doc: Doc) -> list[dict[str, Any]]:
        verbs = []
        fallback_verbs = {
            "go", "went", "gone", "move", "moved", "travel", "travelled", "traveled",
            "arrive", "arrived", "leave", "left", "deport", "deported", "live", "lived",
            "walk", "walked", "hide", "hid", "escape", "escaped", "feel", "felt", "find", "found",
        }
        for tok in doc:
            lower = tok.text.lower()
            is_verb = tok.pos_ in {"VERB", "AUX"} or tok.tag_.startswith("VB")
            if not is_verb and not tok.pos_:
                is_verb = lower in fallback_verbs or lower.endswith("ed")
            if is_verb:
                verbs.append({
                    "text": tok.text,
                    "lemma": tok.lemma_ or lower,
                    "pos": tok.pos_ or "VERB?",
                    "tag": tok.tag_,
                    "start_char": tok.idx,
                    "end_char": tok.idx + len(tok),
                })
        return verbs

    def annotate_texts(
        self,
        texts: Sequence[str],
        file_id: Optional[str] = None,
        start_seg_id: int = 1,
        include_text: bool = False,
        include_entities: bool = True,
        include_verbs: bool = False,
        metadata: Optional[Sequence[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seg_count = len(texts)
        for idx, text in enumerate(texts, start=start_seg_id):
            rec = self.annotate(text, include_entities=include_entities, include_verbs=include_verbs, include_text=include_text)
            rec.update({"file": file_id, "fileId": file_id, "segId": idx, "segCount": seg_count})
            if metadata and idx - start_seg_id < len(metadata):
                rec.update(metadata[idx - start_seg_id])
            records.append(rec)
        return records

    def annotate_file_chunked(
        self,
        path: Union[str, Path],
        n_segments: Optional[int] = None,
        max_chars: Optional[int] = 14000,
        overlap_chars: int = 0,
        encoding: str = "utf-8",
        errors: str = "ignore",
        include_text: bool = True,
        include_entities: bool = True,
        include_verbs: bool = False,
    ) -> list[dict[str, Any]]:
        p = Path(path)
        text = p.read_text(encoding=encoding, errors=errors)
        segments = split_into_segments(text, n_segments=n_segments, nlp=self.nlp, max_chars=max_chars, overlap_chars=overlap_chars)
        return self.annotate_texts(
            segments,
            file_id=p.stem,
            include_text=include_text,
            include_entities=include_entities,
            include_verbs=include_verbs,
        )

    def _resolve_input_files(
        self,
        inputs: Union[str, Path, Sequence[Union[str, Path]]],
        glob_pattern: str = "*.txt",
        recursive: bool = True,
    ) -> Iterator[Path]:
        if isinstance(inputs, (str, Path)):
            candidates = [inputs]
        else:
            candidates = list(inputs)
        for item in candidates:
            p = Path(item)
            if p.is_file():
                yield p
            elif p.is_dir():
                pattern = f"**/{glob_pattern}" if recursive else glob_pattern
                yield from sorted(x for x in p.glob(pattern) if x.is_file())

    def annotate_inputs(
        self,
        inputs: Union[str, Path, Sequence[Union[str, Path]]],
        glob_pattern: str = "*.txt",
        recursive: bool = True,
        encoding: str = "utf-8",
        errors: str = "ignore",
        include_text: bool = True,
        include_entities: bool = True,
        include_verbs: bool = False,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for p in self._resolve_input_files(inputs, glob_pattern, recursive):
            text = p.read_text(encoding=encoding, errors=errors)
            rec = self.annotate(text, include_entities=include_entities, include_verbs=include_verbs, include_text=include_text)
            rec.update({"file": str(p), "fileId": p.stem, "segId": 1, "segCount": 1})
            records.append(rec)
        return records


def _infer_format(path: Union[str, Path], fmt: Optional[str] = None) -> str:
    if fmt:
        return fmt.lower().lstrip(".")
    name = str(path).lower()
    if name.endswith((".jsonl", ".ndjson")):
        return "jsonl"
    if name.endswith(".tsv"):
        return "tsv"
    if name.endswith(".csv"):
        return "csv"
    return "json"


def _table_safe(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else value


def save_annotations(records: Sequence[dict[str, Any]], path: Union[str, Path], fmt: Optional[str] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    kind = _infer_format(p, fmt)
    rows = list(records)
    if kind == "json":
        p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if kind == "jsonl":
        with p.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    if kind not in {"csv", "tsv"}:
        raise ValueError(f"Unsupported output format: {kind}")
    fieldnames = list(dict.fromkeys(STANDARD_COLUMNS + sorted({k for r in rows for k in r})))
    delimiter = "\t" if kind == "tsv" else ","
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _table_safe(row.get(k)) for k in fieldnames})


def _parse_json_cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s or s.lower() in {"none", "nan", "null"}:
        return []
    if s[0] in "[{":
        try:
            return json.loads(s)
        except Exception:
            return value
    return value


def load_annotations(path: Union[str, Path], fmt: Optional[str] = None, ensure_columns: bool = True):
    import pandas as pd

    p = Path(path)
    kind = _infer_format(p, fmt)
    if kind == "jsonl":
        df = pd.read_json(p, lines=True)
    elif kind == "json":
        df = pd.read_json(p)
    elif kind in {"csv", "tsv"}:
        df = pd.read_csv(p, sep="\t" if kind == "tsv" else ",", dtype=str)
        for col in ("entities", "verb_data", "emotion_dist", "themes"):
            if col in df.columns:
                df[col] = df[col].apply(_parse_json_cell)
    else:
        raise ValueError(f"Unsupported format: {kind}")
    if ensure_columns:
        for col in STANDARD_COLUMNS:
            if col not in df.columns:
                df[col] = [[] for _ in range(len(df))] if col in {"entities", "verb_data", "emotion_dist", "themes"} else None
    return df
