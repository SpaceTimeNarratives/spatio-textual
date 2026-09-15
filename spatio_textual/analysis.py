from __future__ import annotations

import re
from typing import Callable, Iterable, Sequence

THEME_KEYWORDS = {
    "movement": {"went", "moved", "travelled", "traveled", "walked", "train", "transport", "route", "arrived", "left"},
    "persecution": {"camp", "ghetto", "deported", "arrested", "guard", "soldier", "nazi", "gestapo"},
    "family": {"mother", "father", "sister", "brother", "child", "children", "uncle", "aunt", "family"},
    "survival": {"survived", "hid", "hiding", "saved", "escape", "escaped", "liberated", "food", "bread"},
    "place-memory": {"home", "village", "town", "city", "street", "house", "school"},
}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def _themes(text: str) -> list[str]:
    words = set(re.findall(r"[A-Za-z']+", (text or "").lower()))
    return [theme for theme, keys in THEME_KEYWORDS.items() if words & keys]


def analyze_records(
    records: Sequence[dict],
    llm_fn: Callable[[dict], dict] | None = None,
    summarize: bool = True,
    explain: bool = True,
    tag_themes: bool = True,
) -> list[dict]:
    """Add simple summaries, affect explanations and theme tags.

    If ``llm_fn`` is supplied it receives each record and can return richer fields.
    """
    out: list[dict] = []
    for rec in records:
        r = dict(rec)
        if llm_fn:
            enriched = llm_fn(r) or {}
            r.update(enriched)
            out.append(r)
            continue
        text = r.get("text") or ""
        if summarize:
            sents = _sentences(text)
            r["summary"] = " ".join(sents[:2])[:500] if sents else text[:500]
        if explain:
            sent = r.get("sentiment_label") or "unknown sentiment"
            emo = r.get("emotion_label") or "unknown emotion"
            ent_count = len(r.get("entities") or [])
            r["interpretation"] = f"This segment has {sent} sentiment and {emo} emotion, with {ent_count} extracted entities."
        if tag_themes:
            r["themes"] = _themes(text)
        out.append(r)
    return out
