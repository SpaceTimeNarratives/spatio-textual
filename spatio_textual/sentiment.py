from __future__ import annotations

"""
sentiment.py — Pluggable sentiment analyzer with a simple rule backend.

Backends:
- "rule": small lexicon-based polarity scorer (no external deps)
- "hf"  : stub hook; you can wire HuggingFace pipelines later
- "llm" : user-supplied function `llm_fn(texts: list[str]) -> list[dict]`

Output per text: {"label": "positive|neutral|negative", "score": float}
"""
from typing import Callable, List, Dict, Optional
import math

# Tiny illustrative lexicon (extend/replace with a real one later)
POS_WORDS = {
    "joy", "happy", "happiness", "relief", "love", "peace", "hope", "safe", "safely", "freedom", "free",
    "reunited", "help", "helped", "support", "protected", "kind", "kindness", "welcomed", "welcome"
}
NEG_WORDS = {
    "fear", "afraid", "terror", "sad", "sadness", "cry", "cried", "anger", "angry", "hate", "hated",
    "disgust", "hunger", "cold", "death", "dead", "killed", "beaten", "sick", "ill", "hurt", "pain",
    "lost", "loss", "lonely", "alone", "danger", "unsafe", "threat", "starved", "starvation"
}


def _rule_score(text: str) -> Dict:
    toks = [t.strip(".,;:!?\"'()[]{} ").lower() for t in text.split()]
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    raw = pos - neg
    # squashed score in [-1,1]
    score = math.tanh(raw / 3.0)
    label = "neutral"
    if score > 0.15:
        label = "positive"
    elif score < -0.15:
        label = "negative"
    return {"label": label, "score": float(score)}


class SentimentAnalyzer:
    def __init__(self, backend: str = "rule", model_name: Optional[str] = None,
                 llm_fn: Optional[Callable[[List[str]], List[Dict]]] = None):
        self.backend = backend
        self.model_name = model_name
        self.llm_fn = llm_fn
        self._pipe = None
        if backend == "hf":
            # Lazy import / init placeholder
            # from transformers import pipeline
            # self._pipe = pipeline("sentiment-analysis", model=model_name or "cardiffnlp/twitter-roberta-base-sentiment-latest")
            raise NotImplementedError("HF backend not yet wired; pass backend='rule' or 'llm'.")

    def predict(self, texts: List[str]) -> List[Dict]:
        if self.backend == "rule":
            return [_rule_score(t) for t in texts]
        if self.backend == "llm":
            if not self.llm_fn:
                raise ValueError("For backend='llm', provide llm_fn(texts)->list[dict].")
            return self.llm_fn(texts)
        if self.backend == "hf":
            if self._pipe is None:
                raise RuntimeError("HF pipeline not initialized.")
            out = self._pipe(texts)
            # Map to our schema
            results = []
            for r in out:
                lab = r.get("label", "neutral").lower()
                if lab.startswith("pos"): lab = "positive"
                if lab.startswith("neg"): lab = "negative"
                results.append({"label": lab, "score": float(r.get("score", 0.0))})
            return results
        raise ValueError(f"Unknown backend: {self.backend}")
