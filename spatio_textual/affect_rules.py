from __future__ import annotations

import re
import time
from typing import Any

from .telemetry import estimate_tokens

EMOTION_LEXICON = {
    "fear": {"afraid", "frightened", "fearful", "scared", "alarmed", "terrified", "fear"},
    "sadness": {"sad", "sorrowful", "downhearted", "mournful", "melancholy", "saddened", "grief"},
    "anger": {"angry", "furious", "resentful", "irritated", "indignant", "enraged", "incensed"},
    "joy": {"joyful", "delighted", "glad", "cheerful", "elated", "overjoyed", "pleased", "joy"},
    "anxiety": {"anxious", "uneasy", "worried", "nervous", "apprehensive", "worry"},
    "despair": {"despondent", "discouraged", "despair", "dejected", "demoralised", "demoralized", "despairing"},
    "gratitude": {"grateful", "thankful", "appreciative", "thanks", "gratitude", "indebted"},
    "surprise": {"surprised", "astonished", "startled", "amazed", "surprise"},
}

POSITIVE = {
    "joyful", "delighted", "glad", "cheerful", "elated", "overjoyed", "pleased",
    "grateful", "thankful", "appreciative", "thanks", "gratitude", "indebted",
    "welcome", "helped", "relief", "relieved", "smiled", "laughed",
}
NEGATIVE = {
    "afraid", "frightened", "fearful", "scared", "alarmed", "terrified",
    "sad", "sorrowful", "downhearted", "mournful", "melancholy", "saddened", "grief",
    "angry", "furious", "resentful", "irritated", "indignant", "enraged", "incensed",
    "anxious", "uneasy", "worried", "nervous", "apprehensive", "worry",
    "despondent", "discouraged", "despair", "dejected", "demoralised", "demoralized", "despairing",
}
NEGATORS = {"not", "never", "no", "neither", "nor"}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", (text or "").lower())


def _is_negated(tokens: list[str], index: int, window: int = 3) -> bool:
    left = tokens[max(0, index - window):index]
    return any(token in NEGATORS for token in left)


def _active_hits(tokens: list[str], words: set[str]) -> list[str]:
    return [token for i, token in enumerate(tokens) if token in words and not _is_negated(tokens, i)]


def classify_affect_rule(text: str) -> dict[str, Any]:
    """Transparent lexicon-based affect baseline.

    The rule intentionally uses only affect-bearing lexical cues. Historical-domain
    terms such as ``camp`` and ``ghetto`` are not sentiment/emotion evidence.
    Contextual/implicit affect therefore remains outside this baseline's reach.
    """
    start = time.perf_counter()
    tokens = _tokens(text)
    emotion_hits = {
        label: _active_hits(tokens, words)
        for label, words in EMOTION_LEXICON.items()
    }
    emotions = sorted(label for label, hits in emotion_hits.items() if hits)

    pos_hits = _active_hits(tokens, POSITIVE)
    neg_hits = _active_hits(tokens, NEGATIVE)
    if pos_hits and neg_hits:
        sentiment = "mixed"
    elif pos_hits:
        sentiment = "positive"
    elif neg_hits:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    active_words = set(pos_hits + neg_hits)
    for hits in emotion_hits.values():
        active_words.update(hits)
    evidence_quote = None
    if active_words:
        # Rules are token-triggered rather than span-rationale models. The whole
        # source is retained as auditable evidence when at least one cue fired.
        evidence_quote = text

    latency_ms = round((time.perf_counter() - start) * 1000, 3)
    return {
        "sentiment_label": sentiment,
        "emotion_labels": emotions,
        "evidence_quote": evidence_quote,
        "explicit_or_inferred": "explicit" if emotions else "none",
        "requires_review": False,
        "rule_hits": {
            "positive": pos_hits,
            "negative": neg_hits,
            "emotions": emotion_hits,
        },
        "telemetry": {
            "task": "affect",
            "backend": "rule",
            "provider": "local",
            "model": "spatio-textual-affect-lexicon-v1",
            "latency_ms": latency_ms,
            "input_chars": len(text or ""),
            "input_tokens_est": estimate_tokens(text),
            "output_tokens_est": 0,
            "cost_usd_est": 0.0,
            "success": True,
            "error": None,
        },
    }
