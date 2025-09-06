from __future__ import annotations

"""
analysis.py — Lightweight analysis/interpretation helpers with optional LLM hook.

- summarize segments
- explain sentiment/emotion labels
- tag themes via simple keyword buckets (can be swapped for LLM)

Pass `llm_fn(prompt:str)->str` to enable richer outputs without hard-coding providers.
"""
from typing import Callable, Dict, Iterable, List, Optional
import textwrap

# simple theme keywords; extend as needed
THEME_BUCKETS = {
    "family": ["mother", "father", "sister", "brother", "parents", "children", "family"],
    "place": ["camp", "ghetto", "city", "village", "forest", "Amsterdam", "Auschwitz"],
    "movement": ["deported", "transported", "moved", "taken", "arrived", "left"],
}


def _simple_summary(text: str, max_chars: int = 240) -> str:
    return (text[: max_chars - 1] + "…") if len(text) > max_chars else text


def _simple_explain(sentiment_label: Optional[str], emotion_label: Optional[str]) -> str:
    bits = []
    if sentiment_label:
        bits.append(f"Sentiment suggests *{sentiment_label}* tone.")
    if emotion_label and emotion_label != "neutral":
        bits.append(f"Dominant emotion appears to be *{emotion_label}*.")
    if not bits:
        bits.append("No strong affective signal detected.")
    return " ".join(bits)


def _simple_themes(text: str) -> List[str]:
    t = text.lower()
    out = []
    for theme, kws in THEME_BUCKETS.items():
        if any(kw.lower() in t for kw in kws):
            out.append(theme)
    return out


def analyze_records(records: List[Dict], *, llm_fn: Optional[Callable[[str], str]] = None,
                    summarize: bool = True, explain: bool = True, tag_themes: bool = True,
                    max_chars: int = 240) -> List[Dict]:
    """
    Enrich records with `summary`, `interpretation`, and `themes`.
    If `llm_fn` is provided, it will be used for the summary+interpretation of each record.
    """
    out: List[Dict] = []
    for r in records:
        text = r.get("text") or r.get("segment") or r.get("raw") or ""
        sentiment_label = r.get("sentiment_label")
        emotion_label = r.get("emotion_label")

        summary = None
        interpretation = None

        if llm_fn is not None:
            prompt = textwrap.dedent(f"""
            Summarize the following testimony segment in one sentence. Then, in a second sentence,
            explain how its emotional tone relates to any detected sentiment/emotion labels.

            TEXT:\n{text}
            SENTIMENT: {sentiment_label}
            EMOTION: {emotion_label}
            """)
            try:
                resp = llm_fn(prompt)
                if isinstance(resp, str):
                    # naive split
                    parts = [p.strip() for p in resp.split("\n") if p.strip()]
                    if parts:
                        summary = parts[0]
                    if len(parts) > 1:
                        interpretation = parts[1]
            except Exception:
                pass  # fall back to simple

        if summary is None and summarize:
            summary = _simple_summary(text, max_chars=max_chars)
        if interpretation is None and explain:
            interpretation = _simple_explain(sentiment_label, emotion_label)

        if tag_themes:
            r["themes"] = _simple_themes(text)
        if summarize:
            r["summary"] = summary
        if explain:
            r["interpretation"] = interpretation

        out.append(r)
    return out
