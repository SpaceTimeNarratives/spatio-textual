from __future__ import annotations

import math
import re
import time
from functools import lru_cache
from typing import Any, Callable, Iterable, Optional

from .llm import LLMClient
from .telemetry import estimate_tokens

EMOTIONS = ["Neutral", "Joy", "Surprise", "Sadness", "Fear", "Anger", "Disgust"]
LEXICON = {
    "Joy": {"happy", "joy", "glad", "relief", "relieved", "safe", "saved", "free", "liberated", "hope"},
    "Surprise": {"suddenly", "unexpected", "surprised", "shock", "shocked", "astonished"},
    "Sadness": {"sad", "cry", "crying", "lost", "death", "died", "dead", "alone", "grief", "mourning"},
    "Fear": {"fear", "afraid", "terrified", "panic", "hiding", "threat", "danger", "camp", "deported"},
    "Anger": {"angry", "anger", "rage", "furious", "hate", "hated", "unfair"},
    "Disgust": {"disgust", "filthy", "dirty", "sick", "disease", "smell", "stench"},
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", (text or "").lower())


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    m = max(scores.values()) if scores else 0.0
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(exps.values()) or 1.0
    return {k: round(v / z, 4) for k, v in exps.items()}


def _winner(dist: dict[str, float], margin: float = 0.10) -> tuple[str, float]:
    ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return "mixed", 0.0
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < margin:
        return "mixed", ranked[0][1]
    return ranked[0]


@lru_cache(maxsize=8)
def _hf_emotion(model_name: str, model_revision: str | None = None):
    """Load and cache a Hugging Face emotion pipeline with an optional revision pin."""
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install transformer dependencies with: pip install -e '.[transformers]'") from exc
    kwargs: dict[str, Any] = {"model": model_name, "top_k": None}
    if model_revision:
        kwargs["revision"] = model_revision
    return pipeline("text-classification", **kwargs)


class EmotionAnalyzer:
    """Ekman-style emotion distribution with rule, HF and LLM backends."""

    def __init__(
        self,
        backend: str = "rule",
        model_name: Optional[str] = None,
        llm_fn: Optional[Callable[[str], dict]] = None,
        provider: str = "openai",
        mixed_margin: float = 0.10,
        model_revision: Optional[str] = None,
    ):
        self.backend = backend
        self.model_name = model_name or ("j-hartmann/emotion-english-distilroberta-base" if backend == "hf" else "rule")
        self.model_revision = model_revision
        self.llm_fn = llm_fn
        self.provider = provider
        self.mixed_margin = mixed_margin

    def predict(self, texts: Iterable[str]) -> list[dict[str, Any]]:
        if self.backend == "hf":
            return [self._hf(t) for t in texts]
        if self.backend == "llm":
            return [self._llm(t) for t in texts]
        if self.backend == "callback" and self.llm_fn:
            return [self.llm_fn(t) for t in texts]
        return [self._rule(t) for t in texts]

    def explain(self, text: str) -> dict[str, Any]:
        """Return the lexical cues used by the transparent rule backend."""
        if self.backend != "rule":
            raise ValueError("explain() is available only for the rule emotion backend")
        tokens = _tokens(text)
        return {
            "tokens": tokens,
            "matched_terms": {
                emotion: [token for token in tokens if token in words]
                for emotion, words in LEXICON.items()
            },
        }

    def _rule(self, text: str) -> dict[str, Any]:
        start = time.perf_counter()
        toks = _tokens(text)
        scores = {emo: 0.0 for emo in EMOTIONS}
        scores["Neutral"] = max(1.0, len(toks) / 60.0)
        hits: dict[str, int] = {}
        for emo, words in LEXICON.items():
            hits[emo] = sum(1 for t in toks if t in words)
            scores[emo] += hits[emo]
        dist = _softmax(scores)
        label, score = _winner(dist, self.mixed_margin)
        return {
            "label": label,
            "score": round(float(score), 4),
            "distribution": dist,
            "hits": hits,
            "telemetry": {
                "task": "emotion",
                "backend": "rule",
                "provider": "local",
                "model": "ekman-lexicon-v2",
                "model_revision": None,
                "latency_ms": round((time.perf_counter() - start) * 1000, 3),
                "input_chars": len(text or ""),
                "input_tokens_est": estimate_tokens(text),
                "output_tokens_est": estimate_tokens(str(dist)),
                "cost_usd_est": 0.0,
                "success": True,
                "error": None,
            },
        }

    def _hf(self, text: str) -> dict[str, Any]:
        start = time.perf_counter()
        error = None
        dist = {emo: 0.0 for emo in EMOTIONS}
        try:
            raw = _hf_emotion(self.model_name, self.model_revision)(text or "")
            if raw and isinstance(raw[0], list):
                raw = raw[0]
            for item in raw:
                lab = str(item.get("label", "")).lower().capitalize()
                if lab in dist:
                    dist[lab] += float(item.get("score", 0.0))
                elif lab.lower() == "neutral":
                    dist["Neutral"] += float(item.get("score", 0.0))
            total = sum(dist.values()) or 1.0
            dist = {k: round(v / total, 4) for k, v in dist.items()}
        except Exception as exc:
            error = str(exc)
        label, score = _winner(dist, self.mixed_margin)
        return {"label": label, "score": round(float(score), 4), "distribution": dist, "telemetry": self._tel(text, start, "hf", "huggingface_transformers", error)}

    def _llm(self, text: str) -> dict[str, Any]:
        if self.llm_fn:
            return self.llm_fn(text)
        data = LLMClient(provider=self.provider, model=self.model_name).classify_json(
            "emotion", text, EMOTIONS,
            "Classify emotions present in a testimony segment. Use mixed when several emotions are close or co-present.",
        )
        dist = {lab: float(data.get("distribution", {}).get(lab, 0.0)) for lab in EMOTIONS}
        label = str(data.get("label") or _winner(dist, self.mixed_margin)[0])
        return {"label": label, "score": max(dist.values()) if dist else 0.0, "distribution": dist, "explanation": data.get("explanation"), "telemetry": data.get("telemetry")}

    def _tel(self, text: str, start: float, backend: str, provider: str, error: str | None) -> dict[str, Any]:
        return {
            "task": "emotion",
            "backend": backend,
            "provider": provider,
            "model": self.model_name,
            "model_revision": self.model_revision,
            "latency_ms": round((time.perf_counter() - start) * 1000, 3),
            "input_chars": len(text or ""),
            "input_tokens_est": estimate_tokens(text),
            "output_tokens_est": 0,
            "cost_usd_est": 0.0 if backend == "hf" else None,
            "success": error is None,
            "error": error,
        }
