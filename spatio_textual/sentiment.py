from __future__ import annotations

import math
import re
import time
from functools import lru_cache
from typing import Any, Callable, Iterable, Optional

from .llm import LLMClient
from .telemetry import estimate_tokens

LABELS = ["positive", "neutral", "negative"]
POSITIVE = {
    "safe", "saved", "helped", "kind", "hope", "joy", "happy", "relief", "relieved", "free", "liberated", "survived", "warm", "welcome", "peace",
}
NEGATIVE = {
    "fear", "afraid", "terrible", "horrible", "sad", "death", "died", "killed", "lost", "hungry", "pain", "cold", "camp", "deport", "deported", "shoot", "beaten", "cry", "crying",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", (text or "").lower())


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    m = max(scores.values()) if scores else 0.0
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(exps.values()) or 1.0
    return {k: round(v / z, 4) for k, v in exps.items()}


def _winner(dist: dict[str, float], margin: float = 0.12) -> tuple[str, float]:
    ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return "mixed", 0.0
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < margin:
        return "mixed", ranked[0][1]
    return ranked[0]


@lru_cache(maxsize=4)
def _hf_sentiment(model_name: str):
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install transformer dependencies with: pip install -e '.[transformers]'") from exc
    return pipeline("text-classification", model=model_name, top_k=None)


class SentimentAnalyzer:
    """Sentiment classifier returning a positive/neutral/negative distribution."""

    def __init__(
        self,
        backend: str = "rule",
        model_name: Optional[str] = None,
        llm_fn: Optional[Callable[[str], dict]] = None,
        provider: str = "openai",
        mixed_margin: float = 0.12,
    ):
        self.backend = backend
        self.model_name = model_name or ("cardiffnlp/twitter-roberta-base-sentiment-latest" if backend == "hf" else "rule")
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

    def _rule(self, text: str) -> dict[str, Any]:
        start = time.perf_counter()
        toks = _tokens(text)
        pos = sum(1 for t in toks if t in POSITIVE)
        neg = sum(1 for t in toks if t in NEGATIVE)
        # neutral prior avoids over-confident labels in short or mixed segments
        scores = {"positive": float(pos), "negative": float(neg), "neutral": max(1.0, len(toks) / 50.0)}
        dist = _softmax(scores)
        label, score = _winner(dist, self.mixed_margin)
        return {
            "label": label,
            "score": round(float(score), 4),
            "distribution": dist,
            "positive_hits": pos,
            "negative_hits": neg,
            "telemetry": {
                "task": "sentiment",
                "backend": "rule",
                "provider": "local",
                "model": "rule-lexicon-v2",
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
        dist = {lab: 0.0 for lab in LABELS}
        try:
            raw = _hf_sentiment(self.model_name)(text or "")
            if raw and isinstance(raw[0], list):
                raw = raw[0]
            for item in raw:
                lab = str(item.get("label", "")).lower()
                score = float(item.get("score", 0.0))
                if "pos" in lab or lab in {"label_2", "5 stars", "4 stars"}:
                    dist["positive"] += score
                elif "neg" in lab or lab in {"label_0", "1 star", "2 stars"}:
                    dist["negative"] += score
                else:
                    dist["neutral"] += score
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
            "sentiment", text, LABELS,
            "Classify testimony segment sentiment. Use mixed when positive/neutral/negative are close or genuinely blended.",
        )
        dist = {lab: float(data.get("distribution", {}).get(lab, 0.0)) for lab in LABELS}
        label = str(data.get("label") or _winner(dist, self.mixed_margin)[0]).lower()
        return {"label": label, "score": max(dist.values()) if dist else 0.0, "distribution": dist, "explanation": data.get("explanation"), "telemetry": data.get("telemetry")}

    def _tel(self, text: str, start: float, backend: str, provider: str, error: str | None) -> dict[str, Any]:
        return {
            "task": "sentiment",
            "backend": backend,
            "provider": provider,
            "model": self.model_name,
            "latency_ms": round((time.perf_counter() - start) * 1000, 3),
            "input_chars": len(text or ""),
            "input_tokens_est": estimate_tokens(text),
            "output_tokens_est": 0,
            "cost_usd_est": 0.0 if backend == "hf" else None,
            "success": error is None,
            "error": error,
        }
