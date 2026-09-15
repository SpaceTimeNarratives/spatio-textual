from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from .geocode import GeoResolver
from .telemetry import estimate_tokens
from .utils import classify_place

HF_LOC_LABELS = {"LOC", "LOCATION", "GPE", "B-LOC", "I-LOC"}
LABEL_MAP = {"LOC": "GPE", "LOCATION": "GPE", "B-LOC": "GPE", "I-LOC": "GPE"}


@lru_cache(maxsize=8)
def _pipeline(model_name: str, revision: str | None = None):
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install transformer dependencies with: pip install -e '.[transformers]' or pip install -r requirements-transformers.txt") from exc
    kwargs: dict[str, Any] = {
        "model": model_name,
        "aggregation_strategy": "simple",
    }
    if revision:
        kwargs["revision"] = revision
    return pipeline("token-classification", **kwargs)


class HFNERAnnotator:
    def __init__(
        self,
        model_name: str,
        link_places: bool = True,
        resolver: GeoResolver | None = None,
        revision: str | None = None,
    ):
        self.model_name = model_name
        self.revision = revision
        self.link_places = link_places
        self.resolver = resolver or GeoResolver()
        self.pipe = _pipeline(model_name, revision)

    @property
    def model_identifier(self) -> str:
        return f"{self.model_name}@{self.revision}" if self.revision else self.model_name

    def annotate(self, text: str, include_text: bool = True) -> dict[str, Any]:
        start = time.perf_counter()
        rec: dict[str, Any] = {"entities": [], "verb_data": [], "error": None}
        if include_text:
            rec["text"] = text
        try:
            raw = self.pipe(text or "")
            for item in raw:
                label = str(item.get("entity_group") or item.get("entity") or "").replace("LABEL_", "")
                norm_label = LABEL_MAP.get(label, label)
                ent = {
                    "text": item.get("word", ""),
                    "label": norm_label,
                    "start_char": int(item.get("start", 0)),
                    "end_char": int(item.get("end", 0)),
                    "start_token": None,
                    "end_token": None,
                    "place_type": classify_place(item.get("word", ""), norm_label) if norm_label in {"GPE", "LOC", "FAC"} else None,
                    "confidence": round(float(item.get("score", 0.0)), 4),
                    "source": f"hf:{self.model_identifier}",
                    "model_revision": self.revision,
                }
                if self.link_places and ent["place_type"] and ent["place_type"] != "GEONOUN":
                    linked = self.resolver.resolve(ent["text"], ent["label"], context=text)
                    if linked:
                        ent.update(linked)
                rec["entities"].append(ent)
        except Exception as exc:
            rec["error"] = str(exc)
        rec["telemetry"] = [{
            "task": "spatial_entity_recognition",
            "backend": "hf",
            "provider": "huggingface_transformers",
            "model": self.model_identifier,
            "model_revision": self.revision,
            "latency_ms": round((time.perf_counter() - start) * 1000, 3),
            "input_chars": len(text or ""),
            "input_tokens_est": estimate_tokens(text),
            "output_tokens_est": estimate_tokens(str(rec.get("entities", []))),
            "cost_usd_est": 0.0,
            "success": rec.get("error") is None,
            "error": rec.get("error"),
        }]
        return rec
