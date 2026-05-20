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
    seg_start_char: Optional[int] = None
    seg_end_char: Optional[int] = None


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
    labeled: list[tuple[str, str, Optional[int], Optional[int]]] = []
    search_from = 0
    for line in raw_lines:
        line_start = (text or "").find(line, search_from)
        line_end = line_start + len(line) if line_start >= 0 else None
        search_from = line_end or search_from
        role, content = _detect_role(line, patterns)
        content_start = (text or "").find(content, line_start if line_start >= 0 else 0) if content else line_start
        content_end = content_start + len(content) if content_start is not None and content_start >= 0 else line_end
        if join_continuations and labeled and role == "unknown":
            prev_role, prev_text, prev_start, _ = labeled[-1]
            labeled[-1] = (prev_role, f"{prev_text} {content}".strip(), prev_start, content_end)
        else:
            labeled.append((role, content, content_start if content_start >= 0 else None, content_end))
    if not labeled and text.strip():
        labeled = [("unknown", text.strip(), 0, len(text))]

    turns: list[tuple[str, str, Optional[int], Optional[int]]] = []
    for role, content, start, end in labeled:
        if sentence_safe and role == "unknown":
            # Sentence offsets are approximate in this rare branch.
            offset = start or 0
            for sent in _sentencize(content, nlp):
                sent_start = (text or "").find(sent, offset)
                sent_end = sent_start + len(sent) if sent_start >= 0 else None
                turns.append((role, sent, sent_start if sent_start >= 0 else None, sent_end))
                offset = sent_end or offset
        else:
            turns.append((role, content, start, end))

    out: list[Segment] = []
    current_pair: Optional[int] = None
    pair_counter = 0
    for idx, (role, content, start, end) in enumerate(turns, start=1):
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
        out.append(Segment(content, role, idx, is_q, is_answer, qa_id, start, end))
    return out


def segment_testimony_file(path: str | Path, **kwargs) -> list[Segment]:
    p = Path(path)
    return segment_testimony(p.read_text(encoding=kwargs.pop("encoding", "utf-8"), errors=kwargs.pop("errors", "ignore")), **kwargs)
