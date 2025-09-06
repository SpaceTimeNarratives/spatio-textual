from __future__ import annotations

"""
emotion.py — Pluggable emotion classifier with a simple rule backend.

Labels: Neutral, Joy, Surprise, Sadness, Fear, Anger, Disgust
Backends: "rule" (default), "hf", "llm" (like sentiment.py)

Output per text: {"label": <emotion>, "score": float, "distribution": Optional[Dict[label,float]]}
"""
from typing import Callable, Dict, List, Optional
import math

EMO_LEXICON = {
    "joy": {"joy", "happy", "happiness", "relief", "comfort", "grateful", "love"},
    "surprise": {"surprised", "astonished", "unexpected", "suddenly"},
    "sadness": {"sad", "sadness", "cry", "cried", "mourning", "grief", "tears"},
    "fear": {"fear", "afraid", "terrified", "scared", "panic", "threat"},
    "anger": {"anger", "angry", "furious", "rage", "enraged", "resentment"},
    "disgust": {"disgust", "disgusted", "revolted", "nausea", "nauseated"},
}
ALL_LABELS = ["neutral", "joy", "surprise", "sadness", "fear", "anger", "disgust"]


def _rule_emotion(text: str) -> Dict:
    toks = [t.strip(".,;:!?\"'()[]{} ").lower() for t in text.split()]
    scores = {lab: 0 for lab in ALL_LABELS}
    for lab, words in EMO_LEXICON.items():
        scores[lab] = sum(1 for t in toks if t in words)
    total = sum(scores.values())
    if total == 0:
        return {"label": "neutral", "score": 0.0, "distribution": {k: (1.0 if k=="neutral" else 0.0) for k in ALL_LABELS}}
    # Softmax-ish
    dist = {k: math.exp(v) for k, v in scores.items()}
    z = sum(dist.values())
    dist = {k: v / z for k, v in dist.items()}
    top = max((v, k) for k, v in dist.items())[1]
    return {"label": top, "score": float(dist[top]), "distribution": dist}


class EmotionAnalyzer:
    def __init__(self, backend: str = "rule", model_name: Optional[str] = None,
                 llm_fn: Optional[Callable[[List[str]], List[Dict]]] = None):
        self.backend = backend
        self.model_name = model_name
        self.llm_fn = llm_fn
        self._pipe = None
        if backend == "hf":
            raise NotImplementedError("HF backend not yet wired; pass backend='rule' or 'llm'.")

    def predict(self, texts: List[str]) -> List[Dict]:
        if self.backend == "rule":
            return [_rule_emotion(t) for t in texts]
        if self.backend == "llm":
            if not self.llm_fn:
                raise ValueError("For backend='llm', provide llm_fn(texts)->list[dict].")
            return self.llm_fn(texts)
        if self.backend == "hf":
            if self._pipe is None:
                raise RuntimeError("HF pipeline not initialized.")
            out = self._pipe(texts)
            # Expect list of dicts with label/score keys
            return [{"label": r.get("label", "neutral"), "score": float(r.get("score", 0.0))} for r in out]
        raise ValueError(f"Unknown backend: {self.backend}")
