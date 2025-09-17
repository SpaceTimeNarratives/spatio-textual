from __future__ import annotations

"""
sentiment.py — Pluggable sentiment analyzer with rule / hf / llm backends.

Backends:
- "rule": small lexicon-based polarity scorer (no external deps)
- "hf"  : HuggingFace Transformers pipeline, OR an LLMRouter via llm_fn
- "llm" : via spatio_textual.llm.LLMRouter or a user-supplied callable llm_fn

Output per text: {"label": "positive|neutral|negative", "score": float}
"""
from typing import Callable, List, Dict, Optional, Any
import math
import os

# Optional hook: we only import if available
try:
    from .llm import LLMRouter  # your new llm.py
except Exception:  # pragma: no cover
    LLMRouter = None  # type: ignore

# Tiny illustrative lexicon (extend with a real one as needed)
POS_WORDS = {
    "joy", "happy", "happiness", "relief", "love", "peace", "hope", "safe", "safely", "freedom", "free",
    "reunited", "help", "helped", "support", "protected", "kind", "kindness", "welcomed", "welcome"
}
NEG_WORDS = {
    "fear", "afraid", "terror", "sad", "sadness", "cry", "cried", "anger", "angry", "hate", "hated",
    "disgust", "hunger", "cold", "death", "dead", "killed", "beaten", "sick", "ill", "hurt", "pain",
    "lost", "loss", "lonely", "alone", "danger", "unsafe", "threat", "starved", "starvation"
}


def _rule_score(text: str) -> Dict[str, float | str]:
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


def _norm_label(lab: str) -> str:
    lab = (lab or "").strip().lower()
    if lab.startswith("pos"):
        return "positive"
    if lab.startswith("neg"):
        return "negative"
    if lab in {"positive", "neutral", "negative"}:
        return lab
    return "neutral"


class SentimentAnalyzer:
    """
    Parameters
    ----------
    backend : str
        "rule" (default), "hf", or "llm".
    model_name : Optional[str]
        HF model name for backend="hf" (e.g., "cardiffnlp/twitter-roberta-base-sentiment-latest").
        Ignored for "rule". For "llm" you set model via env or llm_fn router.
    llm_fn : Optional[Callable | LLMRouter]
        - If backend="llm":
            * If llm_fn is an LLMRouter, we call llm_fn.sentiment(texts).
            * If llm_fn is a callable, we call llm_fn(texts) and expect [{"label","score"}, ...].
            * If llm_fn is None, we autoconfigure an LLMRouter from env vars.
        - If backend="hf":
            * If llm_fn is an LLMRouter, we also use that path (handy single code path).
            * Else we lazily create a transformers pipeline.
    """
    def __init__(self, backend: str = "rule", model_name: Optional[str] = None,
                 llm_fn: Optional[Callable[[List[str]], List[Dict]] | Any] = None):
        self.backend = backend
        self.model_name = model_name
        self.llm_fn = llm_fn
        self._pipe = None  # HF pipeline cache

    # ------------- helpers -------------

    def _ensure_hf_pipeline(self):
        """Lazily create a transformers sentiment pipeline."""
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline  # lazy import
        except Exception as e:
            raise RuntimeError(
                "Hugging Face Transformers not installed. "
                "Install with: pip install transformers"
            ) from e
        model = self.model_name or "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self._pipe = pipeline("sentiment-analysis", model=model)

    def _ensure_llm_router(self):
        """If llm_fn is None, try to build an LLMRouter from environment variables."""
        if LLMRouter is None:
            raise RuntimeError(
                "LLMRouter not available. Ensure spatio_textual.llm is importable."
            )
        provider = os.getenv("LLM_PROVIDER")
        model = os.getenv("LLM_MODEL")
        if not provider or not model:
            raise ValueError(
                "backend='llm' requires either llm_fn or environment variables "
                "LLM_PROVIDER and LLM_MODEL (e.g., LLM_PROVIDER=openai, LLM_MODEL=gpt-4o-mini)."
            )
        # Optional: pass through common keys; LLMRouter picks up provider-specific env vars.
        base_url = os.getenv("LLM_BASE_URL")  # for openai_compat/xai/together etc.
        return LLMRouter(provider=provider, model=model, base_url=base_url)

    # ------------- API -------------

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Return a list of {"label","score"} items for each text, using the selected backend.
        """
        if self.backend == "rule":
            return [_rule_score(t) for t in texts]

        # --- LLM path (router or callable) ---
        if self.backend == "llm":
            # 1) Router instance supplied
            if LLMRouter is not None and isinstance(self.llm_fn, LLMRouter):
                preds = self.llm_fn.sentiment(texts)  # type: ignore[attr-defined]
                return [{"label": _norm_label(p.get("label")), "score": float(p.get("score", 0.0))} for p in preds]

            # 2) User callable supplied
            if callable(self.llm_fn):
                preds = self.llm_fn(texts)  # type: ignore[call-arg]
                return [{"label": _norm_label(p.get("label")), "score": float(p.get("score", 0.0))} for p in preds]

            # 3) Autoconfigure router from env
            router = self._ensure_llm_router()
            preds = router.sentiment(texts)
            return [{"label": _norm_label(p.get("label")), "score": float(p.get("score", 0.0))} for p in preds]

        # --- HF path ---
        if self.backend == "hf":
            # If caller passed an LLMRouter, allow it here too (single path)
            if LLMRouter is not None and isinstance(self.llm_fn, LLMRouter):
                preds = self.llm_fn.sentiment(texts)  # type: ignore[attr-defined]
                return [{"label": _norm_label(p.get("label")), "score": float(p.get("score", 0.0))} for p in preds]

            # Otherwise use transformers pipeline
            self._ensure_hf_pipeline()
            out = self._pipe(texts)  # type: ignore[misc]
            results = []
            for r in out:
                lab = _norm_label(r.get("label", "neutral"))
                results.append({"label": lab, "score": float(r.get("score", 0.0))})
            return results

        raise ValueError(f"Unknown backend: {self.backend}")
