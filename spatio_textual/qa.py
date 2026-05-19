from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import spacy
from spacy.language import Language


@dataclass
class Segment:
    text: str
    role: str
    turn_id: int
    is_question: bool
    is_answer: bool
    qa_pair_id: Optional[int]


DEFAULT_SPEAKER_PATTERNS: dict[str, list[str]] = {
    "interviewer": [r"^\s*(q|question|interviewer|int\.?|i)\s*[:\-]\s*(.*)$"],
    "witness": [r"^\s*(a|answer|witness|survivor|respondent|r|w)\s*[:\-]\s*(.*)$"],
    "narration": [r"^\s*(narrator|narration|note)\s*[:\-]\s*(.*)$"],
}


def _compile_patterns(patterns: Optional[dict[str, list[str]]] = None) -> dict[str, list[re.Pattern]]:
    return {role: [re.compile(p, flags=re.I) for p in pats] for role, pats in (patterns or DEFAULT_SPEAKER_PATTERNS).items()}


def _detect_role(line: str, patterns: dict[str, list[re.Pattern]]) -> tuple[str, str]:
    for role, pats in patterns.items():
        for pat in pats:
            m = pat.match(line)
            if m:
                content = m.group(2) if len(m.groups()) >= 2 else line[m.end():]
                return role, content.strip()
    return "unknown", line.strip()


def _is_question(text: str) -> bool:
    stripped = text.strip()
    return stripped.endswith("?") or stripped.lower().split(" ", 1)[0] in {"what", "where", "when", "why", "how", "who", "did", "were", "was", "do", "can", "could"}


def _sentencize(text: str, nlp: Optional[Language]) -> list[str]:
    if not text:
        return []
    if nlp is None:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]


def segment_testimony(
    text: str,
    nlp: Optional[Language] = None,
    speaker_patterns: Optional[dict[str, list[str]]] = None,
    join_continuations: bool = True,
    sentence_safe: bool = False,
) -> list[Segment]:
    """Split testimony transcripts into Q/A-aware turn records."""
    patterns = _compile_patterns(speaker_patterns)
    raw_lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
    labeled: list[tuple[str, str]] = []
    for line in raw_lines:
        role, content = _detect_role(line, patterns)
        if join_continuations and labeled and role == "unknown":
            prev_role, prev_text = labeled[-1]
            labeled[-1] = (prev_role, f"{prev_text} {content}".strip())
        else:
            labeled.append((role, content))
    if not labeled and text.strip():
        labeled = [("unknown", text.strip())]

    turns: list[tuple[str, str]] = []
    for role, content in labeled:
        if sentence_safe and role == "unknown":
            for sent in _sentencize(content, nlp):
                turns.append((role, sent))
        else:
            turns.append((role, content))

    out: list[Segment] = []
    current_pair: Optional[int] = None
    pair_counter = 0
    for idx, (role, content) in enumerate(turns, start=1):
        is_q = role == "interviewer" or (role == "unknown" and _is_question(content))
        is_q = is_q and _is_question(content)
        if is_q:
            pair_counter += 1
            current_pair = pair_counter
            qa_id = current_pair
            is_answer = False
        elif current_pair is not None and role in {"witness", "unknown"}:
            qa_id = current_pair
            is_answer = True
            current_pair = None
        else:
            qa_id = None
            is_answer = False
        out.append(Segment(content, role, idx, is_q, is_answer, qa_id))
    return out


def segment_testimony_file(path: str | Path, **kwargs) -> list[Segment]:
    p = Path(path)
    return segment_testimony(p.read_text(encoding=kwargs.pop("encoding", "utf-8"), errors=kwargs.pop("errors", "ignore")), **kwargs)
