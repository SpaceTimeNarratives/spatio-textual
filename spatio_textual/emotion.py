from __future__ import annotations
"""
emotion.py — Pluggable emotion classifier with **rule / hf / llm** backends.

Now mirrors SentimentAnalyzer features:
- HF + LLM backends (not just rule)
- Optional **signed/valence** score in [-1, 1] via `include_signed=True`
- Optional per-class **distribution** via `include_distribution=True`

Labels: neutral, joy, surprise, sadness, fear, anger, disgust
Output (default): {"label": <emotion>, "score": float}
If include_distribution=True → also {"distribution": {label: prob, ...}}
If include_signed=True → also {"signed": float in [-1,1]} (valence proxy)
"""
from typing import Callable, Dict, List, Optional, Any
import math
import os

# Optional hook: try to import shared LLM router
try:
    from .llm import LLMRouter  # type: ignore
except Exception:  # pragma: no cover
    LLMRouter = None  # type: ignore

# ---------------- rule lexicon ----------------
EMO_LEXICON = {
    "joy": {"joy", "happy", "happiness", "relief", "comfort", "grateful", "love"},
    "surprise": {"surprised", "astonished", "unexpected", "suddenly"},
    "sadness": {"sad", "sadness", "cry", "cried", "mourning", "grief", "tears"},
    "fear": {"fear", "afraid", "terrified", "scared", "panic", "threat"},
    "anger": {"anger", "angry", "furious", "rage", "enraged", "resentment"},
    "disgust": {"disgust", "disgusted", "revolted", "nausea", "nauseated"},
}
ALL_LABELS = ["neutral", "joy", "surprise", "sadness", "fear", "anger", "disgust"]

# Valence weights for signed score (tunable)
VALENCE = {
    "neutral": 0.0,
    "joy": 1.0,
    "surprise": 0.2,
    "sadness": -1.0,
    "fear": -0.8,
    "anger": -0.7,
    "disgust": -0.7,
}


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    exp = {k: math.exp(v) for k, v in scores.items()}
    z = sum(exp.values()) or 1.0
    return {k: v / z for k, v in exp.items()}


def _rule_emotion(text: str, *, include_distribution: bool = False, include_signed: bool = False) -> Dict:
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
        # valence = sum(prob * weight)
        val = sum(dist.get(k, 0.0) * VALENCE.get(k, 0.0) for k in ALL_LABELS)
        # clip to [-1,1]
        result["signed"] = max(-1.0, min(1.0, float(val)))
    return result


# ---------------- helpers ----------------

# Map common HF labels to our canonical set
_EMO_MAP = {
    # common forms
    "neutral": "neutral",
    "joy": "joy", "happiness": "joy",
    "surprise": "surprise",
    "sadness": "sadness", "sad": "sadness",
    "fear": "fear",
    "anger": "anger",
    "disgust": "disgust",
    # generic label_i fallbacks (model-dependent; many use 0=neg,1=neu,2=pos in sentiment, but for emotion we leave as-is)
}


def _norm_emotion(label: str) -> str:
    lab = (label or "").strip().lower()
    return _EMO_MAP.get(lab, lab)


def _signed_from_top(label: str, score: float) -> float:
    lab = _norm_emotion(label)
    w = VALENCE.get(lab, 0.0)
    return max(-1.0, min(1.0, float(w) * float(score)))


def _signed_from_distribution(dist: Dict[str, float]) -> float:
    val = sum(float(dist.get(k, 0.0)) * float(VALENCE.get(k, 0.0)) for k in ALL_LABELS)
    return max(-1.0, min(1.0, val))


class EmotionAnalyzer:
    """
    Parameters
    ----------
    backend : str
        "rule" (default), "hf", or "llm".
    model_name : Optional[str]
        HF model name for backend="hf" (e.g., a text-classification model trained for emotions).
    llm_fn : Optional[Callable | LLMRouter]
        - If backend="llm":
            * If llm_fn is an LLMRouter and has `.emotion`, calls that; else raises unless `llm_fn` is callable.
            * If llm_fn is a callable, expects list of dicts with {"label","score"} and optional {"distribution"}.
            * If llm_fn is None, we try to autoconfigure an LLMRouter and call `.emotion` if present.
        - If backend="hf":
            * Lazily create a transformers pipeline ("text-classification").
            * If include_distribution=True, calls with return_all_scores=True and builds a normalized distribution.
    """
    def __init__(self, backend: str = "rule", model_name: Optional[str] = None,
                 llm_fn: Optional[Callable[[List[str]], List[Dict]] | Any] = None):
        self.backend = backend
        self.model_name = model_name
        self.llm_fn = llm_fn
        self._pipe = None  # HF pipeline cache

    # ------------- HF helpers -------------
    def _ensure_hf_pipeline(self):
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline  # lazy import
        except Exception as e:
            raise RuntimeError(
                "Hugging Face Transformers not installed. Install with: pip install transformers"
            ) from e
        model = self.model_name or "SamLowe/roberta-base-go_emotions"
        # use generic text-classification; many emotion models expose this task
        self._pipe = pipeline("text-classification", model=model)

    # ------------- LLM helpers -------------
    def _ensure_llm_router(self):
        if LLMRouter is None:
            raise RuntimeError("LLMRouter not available. Ensure spatio_textual.llm is importable.")
        provider = os.getenv("LLM_PROVIDER")
        model = os.getenv("LLM_MODEL")
        if not provider or not model:
            raise ValueError(
                "backend='llm' requires either llm_fn or environment variables "
                "LLM_PROVIDER and LLM_MODEL (e.g., LLM_PROVIDER=openai, LLM_MODEL=gpt-4o-mini)."
            )
        base_url = os.getenv("LLM_BASE_URL")
        return LLMRouter(provider=provider, model=model, base_url=base_url)

    # ------------- API -------------
    def predict(self, texts: List[str], *, include_distribution: bool = False, include_signed: bool = False) -> List[Dict]:
        """
        Return a list of dicts for each text.
        Default keys: {"label","score"}
        If include_distribution=True: add {"distribution": {label: prob}}
        If include_signed=True: add {"signed": float in [-1,1]} (valence proxy)
        """
        if self.backend == "rule":
            out = [_rule_emotion(t, include_distribution=include_distribution, include_signed=include_signed) for t in texts]
            return out

        if self.backend == "llm":
            # Router path
            if LLMRouter is not None and isinstance(self.llm_fn, LLMRouter):
                if hasattr(self.llm_fn, "emotion"):
                    preds = self.llm_fn.emotion(texts)  # type: ignore[attr-defined]
                else:
                    raise NotImplementedError(
                        "LLMRouter provided does not implement .emotion(texts). Pass llm_fn=callable that returns emotion labels."
                    )
            # Callable path
            elif callable(self.llm_fn):
                preds = self.llm_fn(texts)  # type: ignore[call-arg]
            else:
                # Autoconfigure router from env and require .emotion
                router = self._ensure_llm_router()
                if hasattr(router, "emotion"):
                    preds = router.emotion(texts)
                else:
                    raise NotImplementedError(
                        "Auto-configured LLMRouter has no .emotion(texts). Provide llm_fn callable that returns emotion predictions."
                    )

            # normalize outputs and optionally compute signed from top/distribution
            results: List[Dict] = []
            for p in preds:
                lab = _norm_emotion(str(p.get("label", "neutral")))
                sc = float(p.get("score", 0.0))
                item = {"label": lab, "score": sc}
                dist = p.get("distribution")
                if include_distribution and isinstance(dist, dict):
                    # ensure normalized
                    s = sum(float(v) for v in dist.values()) or 1.0
                    item["distribution"] = {k: float(v)/s for k, v in dist.items()}
                if include_signed:
                    if isinstance(dist, dict):
                        item["signed"] = _signed_from_distribution(item.get("distribution", dist))
                    else:
                        item["signed"] = _signed_from_top(lab, sc)
                results.append(item)
            return results

        if self.backend == "hf":
            self._ensure_hf_pipeline()
            from typing import Any as _Any
            if include_distribution:
                out = self._pipe(texts, return_all_scores=True)  # type: ignore[misc]
                results: List[Dict] = []
                for dist_list in out:
                    # dist_list: list of {label, score}
                    dist = { _norm_emotion(str(d.get("label", "")).lower()): float(d.get("score", 0.0)) for d in dist_list }
                    # normalize
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
                    lab = _norm_emotion(str(r.get("label", "neutral")))
                    sc = float(r.get("score", 0.0))
                    item = {"label": lab, "score": sc}
                    if include_signed:
                        item["signed"] = _signed_from_top(lab, sc)
                    results.append(item)
                return results

        raise ValueError(f"Unknown backend: {self.backend}")
