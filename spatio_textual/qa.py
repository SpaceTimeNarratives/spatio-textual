from __future__ import annotations

"""
qa.py — Q/A-aware segmentation for testimony transcripts.

- Detects speaker turns and question/answer structure using configurable patterns
- Keeps sentences intact (optional), and emits stable fields for downstream ML/DF

Minimal, dependency-light version. You can extend patterns via YAML later.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Tuple
import re

import spacy

# -------- Defaults --------
DEFAULT_SPEAKER_PATTERNS: Dict[str, List[str]] = {
    "interviewer": [r"^(?:interviewer|int\.?|q\.|i\.|q:|i:)[\s-]*", r"^q\s*[:\-]", r"^i\s*[:\-]"],
    "witness": [r"^(?:witness|a\.|answer|resp\.?|w\.|a:|w:)[\s-]*", r"^a\s*[:\-]", r"^w\s*[:\-]"],
    "narration": [r"^(?:narrator|nar\.?|note\.?|stage\.?)[\s-]*"],
}

# A question if it ends with '?' OR starts with a question cue (who/what/when/...) roughly.
QUESTION_CUE = re.compile(r"\b(?:who|what|when|where|why|how|did|do|does|can|could|would|will|were|is|are|was|am)\b",
                          re.I)


@dataclass
class Segment:
    text: str
    role: str  # interviewer|witness|narration|unknown
    turn_id: int
    is_question: bool
    is_answer: bool
    qa_pair_id: Optional[int]


# -------- Helpers --------

def _compile_patterns(spec: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    return {k: [re.compile(p, re.I) for p in v] for k, v in spec.items()}


def _detect_role(line: str, pat: Dict[str, List[re.Pattern]]) -> Tuple[str, str]:
    """Return (role, stripped_text)."""
    t = line.strip()
    for role, pats in pat.items():
        for p in pats:
            m = p.match(t)
            if m:
                return role, t[m.end():].lstrip()
    return "unknown", t


def _is_question(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return t.endswith("?") or bool(QUESTION_CUE.search(t))


# -------- Public API --------

def segment_testimony(text: str,
                      nlp: Optional[spacy.Language] = None,
                      speaker_patterns: Optional[Dict[str, List[str]]] = None,
                      join_continuations: bool = True,
                      sentence_safe: bool = True) -> List[Segment]:
    """
    Segment testimony text into Q/A-aware turns.

    - Detects role via prefix patterns (e.g., "Q:", "A:", "Interviewer:")
    - Marks questions/answers and pairs Q->A (greedy, linear pass)
    - If sentence_safe=True, merges lines but preserves sentence boundaries via spaCy
    """
    patterns = _compile_patterns(speaker_patterns or DEFAULT_SPEAKER_PATTERNS)

    # Split to raw lines first
    raw_lines = [ln.rstrip("\n") for ln in text.splitlines()]
    lines: List[Tuple[str, str]] = []  # (role, content)
    for ln in raw_lines:
        role, content = _detect_role(ln, patterns)
        if not content:  # skip empty after label
            continue
        lines.append((role, content))

    # Optionally join continuations: consecutive same-role lines collapse
    if join_continuations and lines:
        joined: List[Tuple[str, str]] = []
        cur_role, cur_txt = lines[0]
        for role, content in lines[1:]:
            if role == cur_role and content and not content.startswith("("):
                cur_txt = f"{cur_txt} {content}".strip()
            else:
                joined.append((cur_role, cur_txt))
                cur_role, cur_txt = role, content
        joined.append((cur_role, cur_txt))
        lines = joined

    # Sentence-safe segmentation per joined turn
    if sentence_safe:
        _nlp = nlp or spacy.blank("en")
        if "sentencizer" not in _nlp.pipe_names:
            _nlp.add_pipe("sentencizer")
        sent_turns: List[Tuple[str, str]] = []
        for role, content in lines:
            doc = _nlp(content)
            for s in doc.sents:
                t = s.text.strip()
                if t:
                    sent_turns.append((role, t))
        lines = sent_turns

    # Build segments with Q/A flags
    segments: List[Segment] = []
    turn_id = 1
    qa_pair_id = 0
    last_q_index = None

    for role, content in lines:
        is_q = role == "interviewer" and _is_question(content)
        is_a = role in {"witness", "unknown"} and not is_q
        if is_q:
            qa_pair_id += 1
            last_q_index = qa_pair_id
        seg = Segment(
            text=content,
            role=role,
            turn_id=turn_id,
            is_question=is_q,
            is_answer=is_a,
            qa_pair_id=last_q_index if (is_q or is_a) else None,
        )
        segments.append(seg)
        turn_id += 1

    return segments


def segment_testimony_file(path: Path, **kwargs) -> List[Segment]:
    return segment_testimony(Path(path).read_text(encoding="utf-8", errors="ignore"), **kwargs)
