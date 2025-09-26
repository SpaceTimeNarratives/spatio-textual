from __future__ import annotations
"""
emotion.py — Pluggable emotion classifier with rule / hf / llm backends.

Backends
- "rule": lexicon + softmax → label + prob; optional valence in [-1,1]
- "hf"  : HuggingFace pipeline("text-classification")
- "llm" : Vendor-agnostic LLM via llm.py's LLMRouter or a user callable

Extras
- include_distribution=True → {"distribution": {label: prob}}
- include_signed=True → {"signed": float in [-1,1]} (valence proxy)
"""
from typing import Callable, Dict, List, Optional, Any
import math
import os

try:
    from .llm import LLMRouter  # type: ignore
except Exception:  # pragma: no cover
    LLMRouter = None  # type: ignore

# -------- Rule lexicon & valence --------
EMO_LEXICON = {
    "joy": {"joy","happy","happiness","relief","comfort","grateful","love"},
    "surprise": {"surprised","astonished","unexpected","suddenly"},
    "sadness": {"sad","sadness","cry","cried","mourning","grief","tears"},
    "fear": {"fear","afraid","terrified","scared","panic","threat"},
    "anger": {"anger","angry","furious","rage","enraged","resentment"},
    "disgust": {"disgust","disgusted","revolted","nausea","nauseated"},
}
ALL_LABELS = ["neutral","joy","surprise","sadness","fear","anger","disgust"]
VALENCE = {
    "neutral":0.0, "joy":1.0, "surprise":0.2,
    "sadness":-1.0, "fear":-0.8, "anger":-0.7, "disgust":-0.7,
}

def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    exp = {k: math.exp(v) for k, v in scores.items()}
    z = sum(exp.values()) or 1.0
    return {k: v/z for k, v in exp.items()}

def _rule_emotion(text: str, *, include_distribution=False, include_signed=False) -> Dict:
    toks = [t.strip(".,;:!?\"'()[]{} ").lower() for t in text.split()]
    counts = {lab: 0.0 for lab in ALL_LABELS}
    for lab, words in EMO_LEXICON.items():
        counts[lab] = float(sum(1 for t in toks if t in words))
    total = sum(counts.values())
    if total == 0:
        dist = {k: (1.0 if k == "neutral" else 0.0) for k in ALL_LABELS}
        top = "neutral"
    else:
        dist = _softmax(counts)
        top = max(dist, key=dist.get)
    result = {"label": top, "score": float(dist[top])}
    if include_distribution:
        result["distribution"] = dist
    if include_signed:
        val = sum(dist.get(k, 0.0) * VALENCE.get(k, 0.0) for k in ALL_LABELS)
        result["signed"] = max(-1.0, min(1.0, float(val)))
    return result

# -------- Label normalization & valence helpers --------
_EMO_MAP = {
    "neutral":"neutral",
    "joy":"joy","happiness":"joy",
    "surprise":"surprise",
    "sadness":"sadness","sad":"sadness",
    "fear":"fear",
    "anger":"anger",
    "disgust":"disgust",
}
def _norm_emotion(label: str) -> str:
    lab = (label or "").strip().lower()
    return _EMO_MAP.get(lab, lab)

def _signed_from_top(label: str, score: float) -> float:
    w = VALENCE.get(_norm_emotion(label), 0.0)
    return max(-1.0, min(1.0, float(w)*float(score)))

def _signed_from_distribution(dist: Dict[str, float]) -> float:
    val = sum(float(dist.get(k,0.0))*float(VALENCE.get(k,0.0)) for k in ALL_LABELS)
    return max(-1.0, min(1.0, val))

# -------- Analyzer --------
class EmotionAnalyzer:
    """
    backend: "rule" (default) | "hf" | "llm"
    model_name: HF model for backend="hf" (e.g., SamLowe/roberta-base-go_emotions)
    llm_fn: LLMRouter with .emotion(texts) or callable(texts)->[{"label","score","distribution"?}]
    """
    def __init__(self, backend: str = "rule", model_name: Optional[str] = None,
                 llm_fn: Optional[Callable[[List[str]], List[Dict]] | Any] = None):
        self.backend = backend
        self.model_name = model_name
        self.llm_fn = llm_fn
        self._pipe = None

    # HF
    def _ensure_hf_pipeline(self):
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline
        except Exception as e:
            raise RuntimeError("Transformers not installed. pip install transformers") from e
        model = self.model_name or "SamLowe/roberta-base-go_emotions"
        self._pipe = pipeline("text-classification", model=model)

    # LLM
    def _ensure_llm_router(self):
        if LLMRouter is None:
            raise RuntimeError("LLMRouter not available. Ensure spatio_textual.llm is importable.")
        provider = os.getenv("LLM_PROVIDER")
        model = os.getenv("LLM_MODEL")
        if not provider or not model:
            raise ValueError("backend='llm' needs llm_fn or env LLM_PROVIDER + LLM_MODEL.")
        base_url = os.getenv("LLM_BASE_URL")
        return LLMRouter(provider=provider, model=model, base_url=base_url)

    def predict(self, texts: List[str], *, include_distribution: bool = False,
                include_signed: bool = False) -> List[Dict]:
        # Rule
        if self.backend == "rule":
            return [_rule_emotion(t, include_distribution=include_distribution,
                                  include_signed=include_signed) for t in texts]

        # LLM
        if self.backend == "llm":
            if LLMRouter is not None and isinstance(self.llm_fn, LLMRouter):
                if not hasattr(self.llm_fn, "emotion"):
                    raise NotImplementedError("LLMRouter provided does not implement .emotion(texts).")
                preds = self.llm_fn.emotion(texts)  # type: ignore[attr-defined]
            elif callable(self.llm_fn):
                preds = self.llm_fn(texts)  # type: ignore[call-arg]
            else:
                router = self._ensure_llm_router()
                if not hasattr(router, "emotion"):
                    raise NotImplementedError("Auto-configured LLMRouter has no .emotion(texts).")
                preds = router.emotion(texts)

            results: List[Dict] = []
            for p in preds:
                lab = _norm_emotion(str(p.get("label","neutral")))
                sc = float(p.get("score", 0.0))
                item = {"label": lab, "score": sc}
                dist = p.get("distribution")
                if include_distribution and isinstance(dist, dict):
                    s = sum(float(v) for v in dist.values()) or 1.0
                    item["distribution"] = {k: float(v)/s for k, v in dist.items()}
                if include_signed:
                    if isinstance(dist, dict):
                        item["signed"] = _signed_from_distribution(item.get("distribution", dist))
                    else:
                        item["signed"] = _signed_from_top(lab, sc)
                results.append(item)
            return results

        # HF
        if self.backend == "hf":
            self._ensure_hf_pipeline()
            if include_distribution:
                out = self._pipe(texts, return_all_scores=True)  # type: ignore[misc]
                results: List[Dict] = []
                for dist_list in out:
                    dist = {_norm_emotion(str(d.get("label","")).lower()): float(d.get("score",0.0)) for d in dist_list}
                    s = sum(dist.values()) or 1.0
                    dist = {k: v/s for k, v in dist.items()}
                    top = max(dist, key=dist.get)
                    item = {"label": top, "score": float(dist[top]), "distribution": dist}
                    if include_signed:
                        item["signed"] = _signed_from_distribution(dist)
                    results.append(item)
                return results
            else:
                out = self._pipe(texts)  # type: ignore[misc]
                results: List[Dict] = []
                for r in out:
                    lab = _norm_emotion(str(r.get("label","neutral")))
                    sc = float(r.get("score", 0.0))
                    item = {"label": lab, "score": sc}
                    if include_signed:
                        item["signed"] = _signed_from_top(lab, sc)
                    results.append(item)
                return results

        raise ValueError(f"Unknown backend: {self.backend}")
