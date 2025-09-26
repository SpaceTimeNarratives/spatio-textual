from __future__ import annotations
"""
Event extraction with optional LLM/HF enrichment, semantic de‑duplication, and tqdm progress.

Fixes:
- Reordered dataclass required fields BEFORE defaults to avoid: 
  TypeError: non-default argument 'event' follows default argument

Features:
- spaCy-based extraction of candidate events per sentence (subj + verb‑lemma + obj/pobj)
- File-level dedupe (lexical) + optional semantic dedupe (sentence‑transformers)
- Aggregates sources (segIds), evidence (sentences), persons/places/dates
- Major/minor categorization by frequency + entity presence
- Optional emotion pooling per event (EmotionAnalyzer)
- Optional normalization via HF text2text or LLMRouter (. _chat_once_json)
- Saving to JSON / JSONL / TSV / CSV
- Optional tqdm progress
"""
from dataclasses import dataclass, asdict, field
from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import hashlib
import json
import os
import uuid

import spacy
from spacy.tokens import Doc, Span

from .utils import load_spacy_model

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from .emotion import EmotionAnalyzer  # type: ignore
except Exception:  # pragma: no cover
    EmotionAnalyzer = None  # type: ignore

try:
    from .llm import LLMRouter  # type: ignore
except Exception:  # pragma: no cover
    LLMRouter = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


# ---------------- Data model ----------------
@dataclass
class EventRecord:
    # REQUIRED (no defaults) — must come first
    fileId: str
    eventId: str
    event: str  # canonical short description

    # Defaults below
    sources: List[int] = field(default_factory=list)      # segIds where found
    evidence: List[str] = field(default_factory=list)     # sentences as evidence
    category: str = "minor"                               # "major" or "minor"
    place: Optional[str] = None
    date: Optional[str] = None
    persons: List[str] = field(default_factory=list)      # primary
    others: List[str] = field(default_factory=list)       # secondary
    emotion: Optional[str] = None
    emotion_valence: Optional[float] = None               # [-1, 1]
    confidence: float = 0.5
    count_mentions: int = 1


# ---------------- Helpers ----------------
def _hash_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _canon_event_phrase(sent: Span) -> str:
    """Compose a compact canonical phrase from a sentence (best effort)."""
    root = next((t for t in sent if t.dep_ == "ROOT" and t.pos_ in {"VERB", "AUX"}), None)
    if root is None:
        root = next((t for t in sent if t.pos_ == "VERB"), None)
    if root is None:
        return sent.text.strip()

    subs = [c for c in root.children if c.dep_ in {"nsubj", "nsubjpass"}]
    dobjs = [c for c in root.children if c.dep_ in {"dobj", "obj"}]
    pobj = []
    for prep in (c for c in root.children if c.dep_ == "prep"):
        pobj.extend([gc for gc in prep.children if gc.dep_ == "pobj"])

    def span_text(tokens):
        return " ".join(t.text for t in tokens).strip()

    subj_text = span_text(subs) or ""
    obj_text = span_text(dobjs or pobj) or ""
    v = root.lemma_ or root.text
    if subj_text and obj_text:
        return f"{subj_text} {v} {obj_text}".strip()
    if subj_text:
        return f"{subj_text} {v}".strip()
    if obj_text:
        return f"{v} {obj_text}".strip()
    return f"{v}".strip()


def _collect_entities(sent: Span) -> Tuple[List[str], List[str], List[str]]:
    persons, places, dates = [], [], []
    for ent in sent.ents:
        if ent.label_ == "PERSON":
            persons.append(ent.text)
        elif ent.label_ in {"GPE", "LOC", "FAC"}:
            places.append(ent.text)
        elif ent.label_ in {"DATE", "TIME"}:
            dates.append(ent.text)
    def _dedupe(xs: List[str]) -> List[str]:
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    return _dedupe(persons), _dedupe(places), _dedupe(dates)


# ---------------- HF/LLM enrichment ----------------
class EventNormalizer:
    """Normalize/expand event records using HF (text2text) or LLMRouter.
    mode="none" | "hf" | "llm".
    """
    def __init__(self, mode: str = "none", model_name: Optional[str] = None,
                 llm_router: Optional["LLMRouter"] = None, temperature: float = 0.0):
        self.mode = mode
        self.model_name = model_name or os.getenv("EVENT_HF_MODEL", "google/flan-t5-base")
        self.router = llm_router
        self.temperature = temperature
        self._pipe = None

    def _ensure_hf(self):
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Transformers not installed. pip install transformers") from e
        self._pipe = pipeline("text2text-generation", model=self.model_name)

    def normalize(self, ev: EventRecord) -> EventRecord:
        if self.mode == "none":
            return ev
        prompt = (
            "Normalize this event as JSON with keys: event, place, date, persons (list), others (list).\n"
            "Keep it concise; don't hallucinate. If unknown, keep field null/empty.\n"
            f"Event phrase: {ev.event}\n"
            f"Place: {ev.place or ''}\nDate: {ev.date or ''}\n"
            f"Persons: {', '.join(ev.persons) or ''}\nOthers: {', '.join(ev.others) or ''}"
        )
        try:
            if self.mode == "hf":
                self._ensure_hf()
                out = self._pipe(prompt, max_new_tokens=128)[0]["generated_text"]  # type: ignore
                data = _safe_json(out)
            elif self.mode == "llm":
                if self.router is None:
                    if LLMRouter is None:
                        return ev
                    provider = os.getenv("LLM_PROVIDER"); model = os.getenv("LLM_MODEL"); base_url = os.getenv("LLM_BASE_URL")
                    if provider and model:
                        self.router = LLMRouter(provider=provider, model=model, base_url=base_url)
                if self.router is None:
                    return ev
                user_prompt = (
                    "You are an event normalizer. Return a single JSON object with keys: event, place, date, persons, others.\n"
                    "Do not invent information; copy from the provided text when present.\n" + prompt
                )
                raw = self.router._chat_once_json(user_prompt)  # type: ignore[attr-defined]
                data = _safe_json(raw)
            else:
                return ev
        except Exception:
            return ev

        if not isinstance(data, dict):
            return ev
        ev.event = str(data.get("event", ev.event) or ev.event)
        ev.place = data.get("place", ev.place) or ev.place
        ev.date = data.get("date", ev.date) or ev.date
        persons = data.get("persons") or []
        others = data.get("others") or []
        if isinstance(persons, list):
            for p in persons:
                if isinstance(p, str) and p not in ev.persons:
                    (ev.persons if len(ev.persons) < 5 else ev.others).append(p)
        if isinstance(others, list):
            for p in others:
                if isinstance(p, str) and p not in ev.persons and p not in ev.others:
                    ev.others.append(p)
        return ev


def _safe_json(txt: str):
    try:
        return json.loads(txt)
    except Exception:
        return None


# ---------------- Semantic de‑duplication ----------------
class SemanticDeduper:
    def __init__(self, model: str = "all-MiniLM-L6-v2", threshold: float = 0.82):
        self.threshold = threshold
        self.model_name = model
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers") from e
            self._model = SentenceTransformer(self.model_name)

    def cluster(self, phrases: List[str]) -> List[int]:
        self._ensure()
        import numpy as np  # type: ignore
        from numpy.linalg import norm  # type: ignore
        X = self._model.encode(phrases, normalize_embeddings=True)
        n = len(phrases)
        clusters = [-1] * n
        centroids = []
        for i in range(n):
            x = X[i]
            best, best_sim = -1, -1.0
            for cid, c in enumerate(centroids):
                sim = float(np.dot(x, c))
                if sim > best_sim:
                    best, best_sim = cid, sim
            if best_sim >= self.threshold and best >= 0:
                clusters[i] = best
                centroids[best] = (centroids[best] + x) / norm(centroids[best] + x)
            else:
                clusters[i] = len(centroids)
                centroids.append(x)
        return clusters


# ---------------- Extractor ----------------
class EventExtractor:
    def __init__(self,
                 nlp: Optional[spacy.Language] = None,
                 emotion_analyzer: Optional["EmotionAnalyzer"] = None,
                 normalizer: Optional[EventNormalizer] = None,
                 dedupe: str = "lexical",  # "lexical" | "semantic"
                 semantic_deduper: Optional[SemanticDeduper] = None,
                 use_tqdm: bool = False):
        self.nlp = nlp or load_spacy_model(os.environ.get("SPACY_MODEL", "en_core_web_sm"))
        self.emotion = emotion_analyzer
        self.normalizer = normalizer or EventNormalizer(mode="none")
        self.dedupe = dedupe
        self.sem_deduper = semantic_deduper
        self.use_tqdm = use_tqdm and (tqdm is not None)

    def extract_file_events(self, segments: Iterable[str], file_id: str,
                            *, major_verb_min_mentions: int = 2,
                            include_emotion: bool = True) -> List[EventRecord]:
        seg_list = list(segments)
        emo_preds = None
        if include_emotion and self.emotion is not None:
            emo_preds = self.emotion.predict(seg_list, include_signed=True)

        candidates: Dict[str, EventRecord] = {}
        iterator = tqdm(range(len(seg_list)), desc=f"events:{file_id}") if self.use_tqdm else range(len(seg_list))
        for seg_id in iterator:
            text = seg_list[seg_id]
            if not text or not text.strip():
                continue
            doc: Doc = self.nlp(text)
            for sent in doc.sents:
                canon = _canon_event_phrase(sent)
                if not canon:
                    continue
                persons, places, dates = _collect_entities(sent)
                key = _hash_key(canon.lower())
                if key not in candidates:
                    ev = EventRecord(
                        fileId=file_id,
                        eventId=str(uuid.uuid4()),
                        event=canon,
                        category="minor",
                        place=places[0] if places else None,
                        date=dates[0] if dates else None,
                        persons=persons[:5],
                        others=[],
                        emotion=None,
                        emotion_valence=None,
                        confidence=0.35,
                        count_mentions=0,
                    )
                    candidates[key] = ev
                ev = candidates[key]
                ev.count_mentions += 1
                ev.sources.append(seg_id)
                ev.evidence.append(sent.text)
                if not ev.place and places:
                    ev.place = places[0]
                if not ev.date and dates:
                    ev.date = dates[0]
                for p in persons:
                    if p not in ev.persons and p not in ev.others:
                        (ev.persons if len(ev.persons) < 5 else ev.others).append(p)
                bonus = 0.05 + 0.05*bool(persons) + 0.05*bool(places) + 0.05*bool(dates)
                ev.confidence = min(1.0, ev.confidence + bonus)

        events = list(candidates.values())

        if self.dedupe == "semantic":
            if self.sem_deduper is None:
                self.sem_deduper = SemanticDeduper()
            phrases = [e.event for e in events]
            cluster_ids = self.sem_deduper.cluster(phrases)
            merged: Dict[int, EventRecord] = {}
            for ev, cid in zip(events, cluster_ids):
                if cid not in merged:
                    merged[cid] = ev
                else:
                    m = merged[cid]
                    m.sources.extend(x for x in ev.sources if x not in m.sources)
                    m.evidence.extend(ev.evidence)
                    m.count_mentions += ev.count_mentions
                    if not m.place and ev.place: m.place = ev.place
                    if not m.date and ev.date: m.date = ev.date
                    for p in ev.persons:
                        if p not in m.persons and p not in m.others:
                            (m.persons if len(m.persons) < 5 else m.others).append(p)
                    for p in ev.others:
                        if p not in m.persons and p not in m.others:
                            m.others.append(p)
                    m.confidence = min(1.0, max(m.confidence, ev.confidence) + 0.05)
            events = list(merged.values())

        if events:
            max_mentions = max(e.count_mentions for e in events)
            for e in events:
                if e.count_mentions >= max(major_verb_min_mentions, max(2, int(0.4 * max_mentions))) or (e.persons and e.place):
                    e.category = "major"

        if include_emotion and self.emotion is not None and emo_preds is not None:
            from collections import Counter
            for ev in events:
                signed_vals, labels = [], []
                for sid in ev.sources:
                    if 0 <= sid < len(emo_preds):
                        p = emo_preds[sid]
                        if isinstance(p.get("signed"), (int, float)):
                            signed_vals.append(float(p["signed"]))
                        if p.get("label"):
                            labels.append(str(p["label"]))
                if signed_vals:
                    ev.emotion_valence = max(-1.0, min(1.0, sum(signed_vals) / len(signed_vals)))
                if labels:
                    ev.emotion = Counter(labels).most_common(1)[0][0]

        if self.normalizer and self.normalizer.mode != "none":
            iterator = tqdm(events, desc=f"normalize:{file_id}") if self.use_tqdm else events
            for i, ev in enumerate(iterator):
                before = ev.event
                ev = self.normalizer.normalize(ev)
                if ev.event and ev.event != before:
                    ev.confidence = min(1.0, ev.confidence + 0.05)
                events[i] = ev

        return events

    def extract_events_from_files(self, files: Dict[str, List[str]], *, include_emotion: bool = True) -> List[EventRecord]:
        all_events: List[EventRecord] = []
        iterator = tqdm(files.items(), desc="files") if self.use_tqdm else files.items()
        for file_id, segs in iterator:
            all_events.extend(self.extract_file_events(segs, file_id=file_id, include_emotion=include_emotion))
        return all_events


# ---------------- Saving ----------------

def _to_records(events: List[EventRecord]) -> List[Dict]:
    return [asdict(e) for e in events]


def save_events(events: List[EventRecord], path: str | Path, fmt: Optional[str] = None) -> Path:
    out = Path(path)
    if fmt is None:
        ext = out.suffix.lower()
        fmt = "jsonl" if ext in {".jsonl", ".ndjson"} else ext.lstrip(".") or "json"

    rows = _to_records(events)

    if fmt in {"json", "geojson"}:
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    if fmt in {"jsonl", "ndjson"}:
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return out

    if pd is None:
        header = [
            "fileId","eventId","event","category","place","date","persons","others",
            "emotion","emotion_valence","count_mentions","confidence","sources"
        ]
        lines = ["\t".join(header)]
        for r in rows:
            line = "\t".join([
                str(r.get("fileId","")),
                str(r.get("eventId","")),
                str(r.get("event","")),
                str(r.get("category","")),
                str(r.get("place","") or ""),
                str(r.get("date","") or ""),
                json.dumps(r.get("persons",[]), ensure_ascii=False),
                json.dumps(r.get("others",[]), ensure_ascii=False),
                str(r.get("emotion","") or ""),
                str(r.get("emotion_valence","") or ""),
                str(r.get("count_mentions",0)),
                str(r.get("confidence",0.0)),
                json.dumps(r.get("sources",[]), ensure_ascii=False),
            ])
            lines.append(line)
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    df = pd.DataFrame(rows)
    if fmt == "csv":
        df.to_csv(out, index=False)
    elif fmt == "tsv":
        df.to_csv(out, sep="\t", index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return out