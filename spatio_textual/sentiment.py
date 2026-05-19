from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable, Optional

POSITIVE = {
    "safe", "saved", "helped", "kind", "hope", "joy", "happy", "relief", "free", "liberated", "survived", "warm", "welcome", "peace",
}
NEGATIVE = {
    "fear", "afraid", "terrible", "horrible", "sad", "death", "died", "killed", "lost", "hungry", "pain", "cold", "camp", "deport", "deported", "shoot", "beaten", "cry", "crying",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", (text or "").lower())


class SentimentAnalyzer:
    """Rule sentiment with a clean LLM/HF callback hook."""

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
        if not toks:
            return {"label": "neutral", "score": 0.0, "positive_hits": 0, "negative_hits": 0}
        pos = sum(1 for t in toks if t in POSITIVE)
        neg = sum(1 for t in toks if t in NEGATIVE)
        score = (pos - neg) / max(pos + neg, 1)
        if score > 0.15:
            label = "positive"
        elif score < -0.15:
            label = "negative"
        else:
            label = "neutral"
        return {"label": label, "score": round(float(score), 3), "positive_hits": pos, "negative_hits": neg}
