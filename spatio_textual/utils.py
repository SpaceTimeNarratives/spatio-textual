from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from .geocode import GeoResolver
from .telemetry import estimate_tokens

DEFAULT_RESOURCES_DIR = Path(__file__).parent / "resources"
STANDARD_COLUMNS = [
    "file", "fileId", "segId", "segCount", "segStartChar", "segEndChar", "segTextCharLength",
    "entities", "verb_data", "event_data", "text", "error",
    "role", "turnId", "qaPairId", "isQuestion", "isAnswer",
    "sentiment_label", "sentiment_score", "sentiment_distribution",
    "emotion_label", "emotion_score", "emotion_dist",
    "summary", "interpretation", "themes", "telemetry", "requires_review", "review_notes",
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
FIRST_PERSON = {"i", "me", "we", "us", "my", "our", "myself", "ourselves"}
MOTION_EVENT_LEMMAS = {
    "go", "move", "travel", "arrive", "leave", "deport", "walk", "hide", "escape", "live", "stay", "return",
    "run", "cross", "come", "take", "send", "transfer", "flee", "find", "work", "sleep", "eat", "wait", "meet",
    "remember", "see", "hear", "feel", "fear", "lose", "survive", "liberate",
}


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
    model_name: str = "en_core_web_trf",
    resources_dir: Union[str, Path, None] = None,
    add_entity_ruler: bool = True,
    disable: Optional[Sequence[str]] = None,
) -> Language:
    """Load spaCy once per process and add optional EntityRuler patterns.

    Default is ``en_core_web_trf`` for quality. For tutorials and CI, use
    ``en_core_web_sm`` or allow the safe blank-pipeline fallback.
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
    as_records: bool = False,
) -> list[Any]:
    """Sentence-safe segmentation.

    By default this preserves the original public API and returns ``list[str]``.
    Pass ``as_records=True`` to include ``segStartChar``, ``segEndChar`` and
    ``segTextCharLength`` for export/audit telemetry.
    """
    if not text or not text.strip():
        return []
    _nlp = nlp or _sentencizer()
    doc = _nlp(text)
    sent_items = [(s.text.strip(), s.start_char, s.end_char) for s in doc.sents if s.text.strip()]
    if not sent_items:
        clean = text.strip()
        recs = [{"text": clean, "segStartChar": 0, "segEndChar": len(text), "segTextCharLength": len(clean)}]
        return recs if as_records else [r["text"] for r in recs]

    chunks: list[list[tuple[str, int, int]]] = []
    if n_segments and n_segments > 0:
        n = max(1, min(int(n_segments), len(sent_items)))
        base, extra = divmod(len(sent_items), n)
        pos = 0
        for i in range(n):
            size = base + (1 if i < extra else 0)
            chunks.append(sent_items[pos:pos + size])
            pos += size
    else:
        if not max_chars or max_chars <= 0:
            chunks = [sent_items]
        else:
            cur: list[tuple[str, int, int]] = []
            cur_len = 0
            for item in sent_items:
                sent = item[0]
                sent_len = len(sent) + (1 if cur else 0)
                if cur and cur_len + sent_len > max_chars:
                    chunks.append(cur)
                    if overlap_chars > 0:
                        overlap: list[tuple[str, int, int]] = []
                        olen = 0
                        for old in reversed(cur):
                            if olen + len(old[0]) > overlap_chars and overlap:
                                break
                            overlap.insert(0, old)
                            olen += len(old[0]) + 1
                        cur = overlap[:]
                        cur_len = sum(len(x[0]) + 1 for x in cur)
                    else:
                        cur = []
                        cur_len = 0
                cur.append(item)
                cur_len += sent_len
            if cur:
                chunks.append(cur)
    out = []
    for chunk in chunks:
        if not chunk:
            continue
        seg_text = " ".join(x[0] for x in chunk).strip()
        out.append({
            "text": seg_text,
            "segStartChar": chunk[0][1],
            "segEndChar": chunk[-1][2],
            "segTextCharLength": len(seg_text),
        })
    return out if as_records else [r["text"] for r in out]


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
    if label in {"GPE", "LOC", "FAC", "CITY"}:
        return "PLACE" if label != "CITY" else "CITY"
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
    """High-level entity, place-linking and event/action annotator."""

    def __init__(
        self,
        nlp: Optional[Language] = None,
        resources_dir: Union[str, Path, None] = None,
        model_name: str | None = None,
        link_places: bool = True,
        resolver: GeoResolver | None = None,
    ):
        self.model_name = model_name or "spacy"
        self.nlp = nlp or load_spacy_model(self.model_name if self.model_name != "spacy" else "en_core_web_trf", resources_dir=resources_dir)
        self.resources_dir = Path(resources_dir) if resources_dir else DEFAULT_RESOURCES_DIR
        self.link_places = link_places
        self.resolver = resolver or GeoResolver()

    def annotate(
        self,
        text: str,
        include_entities: bool = True,
        include_verbs: bool = False,
        include_events: bool = True,
        include_text: bool = False,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        rec: dict[str, Any] = {"entities": [], "verb_data": [], "event_data": [], "error": None, "requires_review": False, "review_notes": []}
        if include_text:
            rec["text"] = text
        try:
            doc = self.nlp(text or "")
            if include_entities:
                rec["entities"] = self._entities(doc)
                if any(e.get("ambiguous") or e.get("resolution_status") == "unresolved" for e in rec["entities"]):
                    rec["requires_review"] = True
                    rec["review_notes"].append("One or more place entities are ambiguous or unresolved.")
            if include_verbs:
                rec["verb_data"] = self._verbs(doc)
            if include_events:
                rec["event_data"] = self._events(doc)
        except Exception as exc:
            rec["error"] = str(exc)
            rec["requires_review"] = True
            rec["review_notes"].append(str(exc))
        rec["telemetry"] = [{
            "task": "spatial_entity_recognition",
            "backend": "spacy",
            "provider": "local",
            "model": self.model_name,
            "latency_ms": round((time.perf_counter() - start) * 1000, 3),
            "input_chars": len(text or ""),
            "input_tokens_est": estimate_tokens(text),
            "output_tokens_est": estimate_tokens(str(rec.get("entities", []))),
            "cost_usd_est": 0.0,
            "success": rec.get("error") is None,
            "error": rec.get("error"),
        }]
        return rec

    def _entities(self, doc: Doc) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen = set()
        for ent in doc.ents:
            key = (ent.start_char, ent.end_char, ent.label_)
            if key in seen:
                continue
            seen.add(key)
            start_tok, end_tok = _token_offsets(doc, ent.start_char, ent.end_char)
            label = ent.label_
            place_type = classify_place(ent.text, label) if label in PLACE_LABELS else None
            row = asdict(EntityRecord(
                text=ent.text,
                label=label,
                start_char=ent.start_char,
                end_char=ent.end_char,
                start_token=start_tok,
                end_token=end_tok,
                place_type=place_type,
            ))
            if self.link_places and place_type and place_type != "GEONOUN":
                linked = self.resolver.resolve(ent.text, label, context=doc.text)
                if linked:
                    row.update(linked)
            out.append(row)
        return out

    def _verbs(self, doc: Doc) -> list[dict[str, Any]]:
        verbs = []
        for tok in doc:
            lower = tok.text.lower()
            is_verb = tok.pos_ in {"VERB", "AUX"} or tok.tag_.startswith("VB")
            if not is_verb and not tok.pos_:
                is_verb = lower in MOTION_EVENT_LEMMAS or lower.endswith("ed")
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

    def _events(self, doc: Doc) -> list[dict[str, Any]]:
        """Extract narrator-centred actions/events, not just all verbs.

        Preference is given to verbs whose subject is first-person (I/we/me/us).
        If dependency information is missing, motion/experience verbs are used as
        a transparent fallback and flagged as lower confidence.
        """
        events: list[dict[str, Any]] = []
        for tok in doc:
            lower = tok.text.lower()
            lemma = (tok.lemma_ or lower).lower()
            is_verb = tok.pos_ in {"VERB", "AUX"} or tok.tag_.startswith("VB") or lemma in MOTION_EVENT_LEMMAS or lower in MOTION_EVENT_LEMMAS
            if not is_verb:
                continue
            subjects = [c for c in tok.children if c.dep_ in {"nsubj", "nsubjpass", "agent"}]
            first_person_subject = any(s.text.lower() in FIRST_PERSON for s in subjects)
            # passive/deportation often has narrator as object: "they deported us"
            first_person_object = any(c.text.lower() in FIRST_PERSON and c.dep_ in {"dobj", "obj", "pobj", "iobj"} for c in tok.children)
            fallback_event = not subjects and lemma in MOTION_EVENT_LEMMAS
            if first_person_subject or first_person_object or fallback_event:
                events.append({
                    "event": tok.text,
                    "lemma": lemma,
                    "event_type": "narrator_action" if first_person_subject else "narrator_experience" if first_person_object else "movement_or_experience_fallback",
                    "subject": [s.text for s in subjects],
                    "first_person_subject": first_person_subject,
                    "first_person_object": first_person_object,
                    "start_char": tok.idx,
                    "end_char": tok.idx + len(tok),
                    "confidence": 0.9 if (first_person_subject or first_person_object) else 0.55,
                    "source": "dependency" if (subjects or first_person_object) else "lexical_fallback",
                })
        return events

    def annotate_texts(
        self,
        texts: Sequence[str | dict[str, Any]],
        file_id: Optional[str] = None,
        start_seg_id: int = 1,
        include_text: bool = False,
        include_entities: bool = True,
        include_verbs: bool = False,
        include_events: bool = True,
        metadata: Optional[Sequence[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seg_count = len(texts)
        for offset, item in enumerate(texts):
            idx = start_seg_id + offset
            if isinstance(item, dict):
                text = str(item.get("text", ""))
                seg_meta = {k: item.get(k) for k in ("segStartChar", "segEndChar", "segTextCharLength") if k in item}
            else:
                text = str(item)
                seg_meta = {"segStartChar": None, "segEndChar": None, "segTextCharLength": len(text)}
            rec = self.annotate(text, include_entities=include_entities, include_verbs=include_verbs, include_events=include_events, include_text=include_text)
            rec.update({"file": file_id, "fileId": file_id, "segId": idx, "segCount": seg_count, **seg_meta})
            if metadata and offset < len(metadata):
                rec.update(metadata[offset])
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
        include_events: bool = True,
    ) -> list[dict[str, Any]]:
        p = Path(path)
        text = p.read_text(encoding=encoding, errors=errors)
        segments = split_into_segments(text, n_segments=n_segments, nlp=self.nlp, max_chars=max_chars, overlap_chars=overlap_chars, as_records=True)
        return self.annotate_texts(
            segments,
            file_id=p.stem,
            include_text=include_text,
            include_entities=include_entities,
            include_verbs=include_verbs,
            include_events=include_events,
        )

    def _resolve_input_files(self, inputs: Union[str, Path, Sequence[Union[str, Path]]], glob_pattern: str = "*.txt", recursive: bool = True) -> Iterator[Path]:
        candidates = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
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
        include_events: bool = True,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for p in self._resolve_input_files(inputs, glob_pattern, recursive):
            text = p.read_text(encoding=encoding, errors=errors)
            rec = self.annotate(text, include_entities=include_entities, include_verbs=include_verbs, include_events=include_events, include_text=include_text)
            rec.update({"file": str(p), "fileId": p.stem, "segId": 1, "segCount": 1, "segStartChar": 0, "segEndChar": len(text), "segTextCharLength": len(text)})
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
        for col in ("entities", "verb_data", "event_data", "emotion_dist", "sentiment_distribution", "themes", "telemetry", "review_notes"):
            if col in df.columns:
                df[col] = df[col].apply(_parse_json_cell)
    else:
        raise ValueError(f"Unsupported format: {kind}")
    if ensure_columns:
        for col in STANDARD_COLUMNS:
            if col not in df.columns:
                df[col] = [[] for _ in range(len(df))] if col in {"entities", "verb_data", "event_data", "emotion_dist", "sentiment_distribution", "themes", "telemetry", "review_notes"} else None
    return df
