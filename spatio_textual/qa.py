from __future__ import annotations
"""
qa.py — unified splitter for testimony files

This combines BOTH workflows:
  1) Sentence‑safe chunking into ~N segments (no broken sentences)
  2) Interviewer↔Survivor Q/A extraction (robust to several transcript formats)

Drop‑in: it does NOT require changes elsewhere in spatio_textual.

Key features
------------
- Sentence‑safe segmentation using spaCy's sentencizer (no training needed)
- Greedy packing into ~N segments while preserving sentence boundaries
- Q/A parser that recognizes multiple label styles: "INT:", "Q:", "A:", ALL‑CAPS names, etc.
- Configurable speaker patterns (pass dict, or load YAML profile if PyYAML present)
- Stable fields for downstream processing:
    * segments: {fileId, segId, text}
    * pairs   : {fileId, pairId, q_speaker, q_text, a_speaker, a_text}
- Tiny, dependency‑light; spaCy only. (PyYAML optional for pattern profiles.)

Usage (Python)
--------------
>>> from spatio_textual.qa import chunk_file, qa_from_file
>>> segs = chunk_file('268.txt', n_segments=100)
>>> pairs = qa_from_file('268.txt', interviewer_tags={'INT','Q'}, exclude_tags={'CREW'})

You can feed `segs[i]['text']` or `pairs[i]['a_text']` directly into your
entity, sentiment/emotion, or event extractors.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Tuple
import re
import os

import spacy

# =========================
# Defaults & Profiles
# =========================
# Minimal base patterns; you can extend/override via function args or YAML
DEFAULT_SPEAKER_PATTERNS: Dict[str, List[str]] = {
    # interviewer prefixes
    "interviewer": [
        r"^(?:interviewer|int\.?|q\.?|i\.?)[\s:-]+",  # Interviewer:, INT:, Q:, I:
        r"^q\s*[:\-]",
        r"^i\s*[:\-]",
    ],
    # witness/answer prefixes
    "witness": [
        r"^(?:witness|answer|resp\.?|a\.?|w\.?)[\s:-]+",  # Witness:, A:, W:, Answer:
        r"^a\s*[:\-]",
        r"^w\s*[:\-]",
    ],
    # narration/non‑dialogue
    "narration": [r"^(?:narrator|nar\.?|note\.?|stage\.?)[\s:-]+"],
}

# Heuristic question cue
QUESTION_CUE = re.compile(r"\b(?:who|what|when|where|why|how|did|do|does|can|could|would|will|were|is|are|was|am)\b",
                          re.I)

# Generic ALL‑CAPS speaker label: e.g., "INT:", "MR SMITH:", "ANNE:"
CAPS_SPEAKER_RE = re.compile(r"^([A-Z][A-Z0-9 .\-]{0,30}):\s*(.*)$")


# =========================
# Data structures
# =========================
@dataclass
class Segment:
    fileId: str
    segId: int
    text: str

@dataclass
class QAPair:
    fileId: str
    pairId: int
    q_speaker: str
    q_text: str
    a_speaker: str
    a_text: str


# =========================
# spaCy loading (lightweight)
# =========================
_def_model = os.getenv("SPACY_MODEL", "en_core_web_sm")

def _load_spacy(model: Optional[str] = None) -> spacy.Language:
    name = model or _def_model
    try:
        nlp = spacy.load(name)
    except Exception:
        nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


# =========================
# (A) Sentence‑safe chunking
# =========================

def split_sentences(text: str, nlp: Optional[spacy.Language] = None) -> List[str]:
    nlp = nlp or _load_spacy()
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def pack_segments(sentences: List[str], n_segments: int = 100) -> List[str]:
    """Greedy packing: keep sentences intact; aim for ~n_segments chunks."""
    if n_segments <= 1 or len(sentences) <= 1:
        return [" ".join(sentences)] if sentences else []
    total = len(sentences)
    size = max(1, total // n_segments + (1 if total % n_segments else 0))
    return [
        " ".join(sentences[i:i+size]).strip()
        for i in range(0, total, size)
        if sentences[i:i+size]
    ]


def chunk_text(text: str, file_id: str = "file", n_segments: int = 100,
               model: Optional[str] = None) -> List[dict]:
    nlp = _load_spacy(model)
    sents = split_sentences(text, nlp)
    segs = pack_segments(sents, n_segments=n_segments)
    return [{"fileId": file_id, "segId": i, "text": seg} for i, seg in enumerate(segs)]


def chunk_file(path: str | Path, n_segments: int = 100,
               model: Optional[str] = None, encoding: str = "utf-8") -> List[dict]:
    p = Path(path)
    text = p.read_text(encoding=encoding, errors="ignore")
    return chunk_text(text, file_id=p.stem, n_segments=n_segments, model=model)


# =========================
# (B) Q/A extraction
# =========================

def _compile_patterns(spec: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    return {k: [re.compile(p, re.I) for p in v] for k, v in spec.items()}


def _detect_role(line: str, pat: Dict[str, List[re.Pattern]]) -> tuple[str, str, str]:
    """Return (role, speaker_tag, stripped_text).
    1) try explicit role patterns; 2) fall back to ALL‑CAPS speaker label.
    """
    t = line.strip()
    for role, pats in pat.items():
        for p in pats:
            m = p.match(t)
            if m:
                return role, role.upper(), t[m.end():].lstrip()
    m2 = CAPS_SPEAKER_RE.match(t)
    if m2:
        spk = m2.group(1).strip()
        rest = m2.group(2).strip()
        # heuristic: interviewer keywords; else assume witness
        if spk in {"INT", "INTERVIEWER", "Q"} or spk.startswith("INT"):
            return "interviewer", spk, rest
        return "witness", spk, rest
    return "unknown", "", t


def _is_question(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return t.endswith("?") or bool(QUESTION_CUE.search(t))


def extract_qa_pairs(text: str,
                     interviewer_tags: Iterable[str] | None = None,
                     exclude_tags: Iterable[str] | None = None,
                     speaker_patterns: Optional[Dict[str, List[str]]] = None,
                     merge_multiturn: bool = True,
                     sentence_safe: bool = True,
                     file_id: str = "file") -> List[dict]:
    """Build Q/A pairs.
    Q = contiguous interviewer turns (merged)
    A = contiguous turns by the first non‑interviewer speaker right after Q
    """
    patterns = _compile_patterns(speaker_patterns or DEFAULT_SPEAKER_PATTERNS)
    iv_tags = set((interviewer_tags or {"INT", "INTERVIEWER", "Q"}))
    ex_tags = set((exclude_tags or {"CREW"}))

    raw_lines = [ln.rstrip("\n") for ln in text.splitlines()]

    # Identify role/speaker, strip labels
    labeled: List[tuple[str,str,str]] = []  # (role, speaker_tag, text)
    for ln in raw_lines:
        role, tag, content = _detect_role(ln, patterns)
        if not content:
            continue
        if tag in ex_tags:
            continue
        labeled.append((role, tag or role.upper(), content))

    # Optionally merge continuations of same role+speaker
    if merge_multiturn and labeled:
        merged: List[tuple[str,str,str]] = []
        cur = list(labeled[0])  # mutable copy
        for role, tag, content in labeled[1:]:
            if role == cur[0] and tag == cur[1] and content and not content.startswith("("):
                cur[2] = f"{cur[2]} {content}".strip()
            else:
                merged.append(tuple(cur))
                cur = [role, tag, content]
        merged.append(tuple(cur))
        labeled = merged

    # Sentence‑safe split per turn
    if sentence_safe:
        nlp = _load_spacy()
        sent_turns: List[tuple[str,str,str]] = []
        for role, tag, content in labeled:
            for s in nlp(content).sents:
                t = s.text.strip()
                if t:
                    sent_turns.append((role, tag, t))
        labeled = sent_turns

    # Build Q→A pairs
    pairs: List[QAPair] = []
    pid = 0
    i = 0
    while i < len(labeled):
        role, tag, txt = labeled[i]
        if (role == "interviewer" or tag in iv_tags) and _is_question(txt):
            # collect all interviewer question lines
            q_chunks = [txt]
            i += 1
            while i < len(labeled) and (labeled[i][0] == "interviewer" or labeled[i][1] in iv_tags):
                q_chunks.append(labeled[i][2])
                i += 1
            # find first answer speaker (not interviewer)
            if i >= len(labeled):
                break
            a_role, a_tag, a_txt = labeled[i]
            if a_role == "interviewer" or a_tag in iv_tags:
                i += 1
                continue
            a_speaker = a_tag or a_role.upper()
            a_chunks = [a_txt]
            i += 1
            if merge_multiturn:
                while i < len(labeled) and labeled[i][1] == a_speaker:
                    a_chunks.append(labeled[i][2])
                    i += 1
            pid += 1
            pairs.append(QAPair(
                fileId=file_id,
                pairId=pid,
                q_speaker="INTERVIEWER",
                q_text=" ".join(q_chunks).strip(),
                a_speaker=a_speaker,
                a_text=" ".join(a_chunks).strip(),
            ))
        else:
            i += 1

    return [vars(p) for p in pairs]


def qa_from_file(path: str | Path,
                 interviewer_tags: Iterable[str] | None = None,
                 exclude_tags: Iterable[str] | None = None,
                 speaker_patterns: Optional[Dict[str, List[str]]] = None,
                 merge_multiturn: bool = True,
                 sentence_safe: bool = True,
                 encoding: str = "utf-8") -> List[dict]:
    p = Path(path)
    text = p.read_text(encoding=encoding, errors="ignore")
    return extract_qa_pairs(text,
                            interviewer_tags=interviewer_tags,
                            exclude_tags=exclude_tags,
                            speaker_patterns=speaker_patterns,
                            merge_multiturn=merge_multiturn,
                            sentence_safe=sentence_safe,
                            file_id=p.stem)


# =========================
# Optional: save helpers (pandas if available)
# =========================
try:
    import pandas as _pd  # type: ignore
except Exception:
    _pd = None  # type: ignore


def save_segments(segments: List[dict], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if _pd is None or out.suffix.lower() == ".jsonl":
        with out.open("w", encoding="utf-8") as f:
            for r in segments:
                import json
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        _pd.DataFrame(segments).to_csv(out, index=False)
    return out


def save_pairs(pairs: List[dict], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if _pd is None or out.suffix.lower() == ".jsonl":
        with out.open("w", encoding="utf-8") as f:
            for r in pairs:
                import json
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        _pd.DataFrame(pairs).to_csv(out, index=False)
    return out
