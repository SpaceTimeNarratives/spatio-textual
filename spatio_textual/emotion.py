from __future__ import annotations

import re
from typing import Callable, Iterable, Optional

EMOTIONS = ["Neutral", "Joy", "Surprise", "Sadness", "Fear", "Anger", "Disgust"]
LEXICON = {
    "Joy": {"happy", "joy", "glad", "relief", "relieved", "safe", "saved", "free", "liberated", "hope"},
    "Surprise": {"suddenly", "unexpected", "surprised", "shock", "shocked", "astonished"},
    "Sadness": {"sad", "cry", "crying", "lost", "death", "died", "dead", "alone", "grief", "mourning"},
    "Fear": {"fear", "afraid", "terrified", "panic", "hiding", "threat", "danger", "camp", "deported"},
    "Anger": {"angry", "anger", "rage", "furious", "hate", "hated", "unfair"},
    "Disgust": {"disgust", "filthy", "dirty", "disease", "stench", "rotting"},
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", (text or "").lower())


class EmotionAnalyzer:
    """Ekman-style rule emotion classifier with LLM/HF hook."""

    def __init__(self, backend: str = "rule", model_name: Optional[str] = None, llm_fn: Optional[Callable[[str], dict]] = None):
        self.backend = backend
        self.model_name = model_name
        self.llm_fn = llm_fn

    def predict(self, texts: Iterable[str]) -> list[dict]:
        if self.backend in {"llm", "hf"} and self.llm_fn:
            return [self.llm_fn(t) for t in texts]
        return [self._rule(t) for t in texts]

    def _rule(self, text: str) -> dict:
        toks = _tokens(text)
        counts = {emo: sum(1 for t in toks if t in words) for emo, words in LEXICON.items()}
        total = sum(counts.values())
        if total == 0:
            dist = {emo: (1.0 if emo == "Neutral" else 0.0) for emo in EMOTIONS}
            return {"label": "Neutral", "score": 1.0, "distribution": dist}
        winner = max(counts, key=counts.get)
        dist = {"Neutral": 0.0}
        dist.update({emo: round(count / total, 3) for emo, count in counts.items()})
        return {"label": winner, "score": round(counts[winner] / total, 3), "distribution": dist}
