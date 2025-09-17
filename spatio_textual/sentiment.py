from __future__ import annotations
"""
sentiment.py — Pluggable sentiment analyser with rule/hf/ llm backends.

Now supports optional **signed scores in [-1,1]** via `include_signed=True` in `predict()`.

Backends:
- "rule": small lexicon-based polarity scorer (no external deps)
- "hf"  : HuggingFace Transformers pipeline, OR an LLMRouter via llm_fn
- "llm" : via spatio_textual.llm.LLMRouter or a user-supplied callable llm_fn

Output per text (default): {"label": "positive|neutral|negative", "score": float}
If `include_signed=True`, adds {"signed": float in [-1,1]}.
"""
from typing import Callable, List, Dict, Optional, Any
import math
import os

# Optional hook: only import if available
try:
    from .llm import LLMRouter  # your llm.py
except Exception:  # pragma: no cover
    LLMRouter = None  # type: ignore

# --- tiny illustrative lexicon ---
POS_WORDS = {
    "joy", "happy", "happiness", "relief", "love", "peace", "hope", "safe", "safely", "freedom", "free",
    "reunited", "help", "helped", "support", "protected", "kind", "kindness", "welcomed", "welcome"
}
NEG_WORDS = {
    "fear", "afraid", "terror", "sad", "sadness", "cry", "cried", "anger", "angry", "hate", "hated",
    "disgust", "hunger", "cold", "death", "dead", "killed", "beaten", "sick", "ill", "hurt", "pain",
    "lost", "loss", "lonely", "alone", "danger", "unsafe", "threat", "starved", "starvation"
}


# ---------------- rule backend ----------------

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


# ---------------- helpers ----------------

def _norm_label(lab: str) -> str:
    lab = (lab or "").strip().lower()
    if lab.startswith("pos"):
        return "positive"
    if lab.startswith("neg"):
        return "negative"
    if lab in {"positive", "neutral", "negative"}:
        return lab
    return "neutral"

# Some HF models use explicit labels; others (e.g., cardiffnlp) use LABEL_0/1/2
_LABEL_MAP = {
    "positive": "pos", "negative": "neg", "neutral": "neu",
    "label_2": "pos", "label_1": "neu", "label_0": "neg",
}

def _signed_from_all_scores(all_scores: List[Dict[str, float]], normalize: bool = True) -> float:
    """Map a pipeline(return_all_scores=True) output (list of {label,score}) to s in [-1,1]."""
    p_pos = p_neg = 0.0
    for d in all_scores:
        lab = _LABEL_MAP.get(str(d.get("label", "")).lower(), str(d.get("label", "")).lower())
        if lab in ("pos", "positive"):
            p_pos = float(d.get("score", 0.0))
        elif lab in ("neg", "negative"):
            p_neg = float(d.get("score", 0.0))
    if normalize:
        denom = max(p_pos + p_neg, 1e-9)
        return (p_pos - p_neg) / denom
    return p_pos - p_neg


def _signed_from_top(label: str, score: float) -> float:
    lab = (label or "").lower()
    if lab.startswith("pos"):
        return float(score)
    if lab.startswith("neg"):
        return -float(score)
    return 0.0


class SentimentAnalyzer:
    """
    Parameters
    ----------
    backend: str
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

    # ------------- HF helpers -------------
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

    # ------------- LLM helpers -------------
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
        base_url = os.getenv("LLM_BASE_URL")  # for openai_compat/xai/together etc.
        return LLMRouter(provider=provider, model=model, base_url=base_url)

    # ------------- API -------------
    def predict(self, texts: List[str], *, include_signed: bool = False, normalize_signed: bool = True) -> List[Dict]:
        """
        Return a list of dicts for each text.
        Default keys: {"label","score"}
        If include_signed=True, also adds {"signed"} in [-1,1].
        """
        if self.backend == "rule":
            out = [_rule_score(t) for t in texts]
            if include_signed:
                # rule "score" is already in [-1,1]; mirror it to "signed"
                for r in out:
                    r["signed"] = float(r["score"])  # identical scale
            return out

        # --- LLM path (router or callable) ---
        if self.backend == "llm":
            # 1) Router instance supplied
            if LLMRouter is not None and isinstance(self.llm_fn, LLMRouter):
                preds = self.llm_fn.sentiment(texts)  # type: ignore[attr-defined]
            # 2) User callable supplied
            elif callable(self.llm_fn):
                preds = self.llm_fn(texts)  # type: ignore[call-arg]
            else:
                # 3) Autoconfigure router from env
                router = self._ensure_llm_router()
                preds = router.sentiment(texts)
            results = [{"label": _norm_label(p.get("label")), "score": float(p.get("score", 0.0))} for p in preds]
            if include_signed:
                for r in results:
                    # signed from top-1 label/score
                    r["signed"] = _signed_from_top(r["label"], r["score"]) if isinstance(r["label"], str) else 0.0
            return results

        # --- HF path ---
        if self.backend == "hf":
            # If caller passed an LLMRouter, allow it here too
            if LLMRouter is not None and isinstance(self.llm_fn, LLMRouter):
                preds = self.llm_fn.sentiment(texts)  # type: ignore[attr-defined]
                results = [{"label": _norm_label(p.get("label")), "score": float(p.get("score", 0.0))} for p in preds]
                if include_signed:
                    for r in results:
                        r["signed"] = _signed_from_top(r["label"], r["score"]) if isinstance(r["label"], str) else 0.0
                return results

            # Otherwise use transformers pipeline
            self._ensure_hf_pipeline()
            if include_signed:
                # Need per-class probabilities to build signed → request full distribution
                out = self._pipe(texts, return_all_scores=True)  # type: ignore[misc]
                results: List[Dict] = []
                for dist in out:
                    s_val = _signed_from_all_scores(dist, normalize=normalize_signed)
                    # get top label for convenience
                    top = max(dist, key=lambda d: float(d.get("score", 0.0)))
                    results.append({
                        "label": _norm_label(str(top.get("label", "neutral"))),
                        "score": float(top.get("score", 0.0)),
                        "signed": float(s_val),
                    })
                return results
            else:
                out = self._pipe(texts)  # type: ignore[misc]
                results = []
                for r in out:
                    lab = _norm_label(r.get("label", "neutral"))
                    results.append({"label": lab, "score": float(r.get("score", 0.0))})
                return results

        raise ValueError(f"Unknown backend: {self.backend}")
