from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from .telemetry import estimate_tokens

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMOTION_REVISION = "0e1cd914e3d46199ed785853e12b57304e04178b"

# Hartmann's native ontology is anger, disgust, fear, joy, neutral, sadness,
# surprise. The package task ontology excludes disgust and does not treat neutral
# as an emotion. Anxiety, despair and gratitude are therefore outside this
# transformer's representational ontology and are reported as such.
REPRESENTABLE_EMOTIONS = ("anger", "fear", "joy", "sadness", "surprise")
UNSUPPORTED_EMOTIONS = ("anxiety", "despair", "gratitude")

# Frozen post-processing settings. The sentiment model has three native labels;
# 'mixed' is emitted only when positive and negative probabilities are both
# substantial and close. The emotion model is single-label softmax; the low
# threshold permits rare secondary labels but is not presented as a native
# multilabel classifier.
MIXED_MIN_PROBABILITY = 0.25
MIXED_MAX_GAP = 0.20
EMOTION_THRESHOLD = 0.25
EMOTION_TOP_FALLBACK = 0.50


def _require_transformers():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install transformer dependencies with: pip install -e '.[transformers]'") from exc
    return AutoModelForSequenceClassification, AutoTokenizer, pipeline


@lru_cache(maxsize=4)
def _classifier(model_name: str, revision: str):
    AutoModel, AutoTokenizer, pipeline = _require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)


def _flat_scores(raw: Any) -> list[dict[str, Any]]:
    if raw and isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    return raw if isinstance(raw, list) else []


def _sentiment_distribution(text: str) -> dict[str, float]:
    raw = _flat_scores(_classifier(SENTIMENT_MODEL, SENTIMENT_REVISION)(text or ""))
    dist = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for item in raw:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        if "pos" in label or label == "label_2":
            dist["positive"] += score
        elif "neg" in label or label == "label_0":
            dist["negative"] += score
        else:
            dist["neutral"] += score
    total = sum(dist.values()) or 1.0
    return {key: value / total for key, value in dist.items()}


def _sentiment_label(dist: dict[str, float]) -> str:
    positive = dist["positive"]
    negative = dist["negative"]
    if (
        min(positive, negative) >= MIXED_MIN_PROBABILITY
        and abs(positive - negative) <= MIXED_MAX_GAP
    ):
        return "mixed"
    return max(dist, key=dist.get)


def _emotion_distribution(text: str) -> dict[str, float]:
    raw = _flat_scores(_classifier(EMOTION_MODEL, EMOTION_REVISION)(text or ""))
    return {str(item.get("label", "")).lower(): float(item.get("score", 0.0)) for item in raw}


def _emotion_labels(dist: dict[str, float]) -> list[str]:
    labels = [
        label
        for label in REPRESENTABLE_EMOTIONS
        if dist.get(label, 0.0) >= EMOTION_THRESHOLD
    ]
    if labels:
        return sorted(labels)
    supported_scores = {label: dist.get(label, 0.0) for label in REPRESENTABLE_EMOTIONS}
    if not supported_scores:
        return []
    top_label = max(supported_scores, key=supported_scores.get)
    top_score = supported_scores[top_label]
    neutral_score = dist.get("neutral", 0.0)
    if top_score >= EMOTION_TOP_FALLBACK and top_score > neutral_score:
        return [top_label]
    return []


def classify_affect_transformer(text: str) -> dict[str, Any]:
    start = time.perf_counter()
    sentiment_dist = _sentiment_distribution(text)
    emotion_dist = _emotion_distribution(text)
    sentiment = _sentiment_label(sentiment_dist)
    emotions = _emotion_labels(emotion_dist)
    latency_ms = round((time.perf_counter() - start) * 1000, 3)
    return {
        "sentiment_label": sentiment,
        "emotion_labels": emotions,
        "evidence_quote": None,
        "explicit_or_inferred": "not_available",
        "requires_review": False,
        "sentiment_distribution": {k: round(v, 6) for k, v in sentiment_dist.items()},
        "emotion_distribution": {k: round(v, 6) for k, v in emotion_dist.items()},
        "representable_emotion_labels": list(REPRESENTABLE_EMOTIONS),
        "unsupported_emotion_labels": list(UNSUPPORTED_EMOTIONS),
        "telemetry": {
            "task": "affect",
            "backend": "transformer",
            "provider": "huggingface_transformers",
            "model": {
                "sentiment": SENTIMENT_MODEL,
                "sentiment_revision": SENTIMENT_REVISION,
                "emotion": EMOTION_MODEL,
                "emotion_revision": EMOTION_REVISION,
            },
            "latency_ms": latency_ms,
            "input_chars": len(text or ""),
            "input_tokens_est": estimate_tokens(text),
            "output_tokens_est": 0,
            "cost_usd_est": 0.0,
            "success": True,
            "error": None,
        },
    }
