from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable, Sequence

from .utils import Annotator, load_spacy_model


@dataclass
class ModelAnnotation:
    model: str
    record: dict[str, Any]


@dataclass
class AdjudicationResult:
    consensus: dict[str, Any]
    disagreements: list[dict[str, Any]]
    votes: dict[str, list[dict[str, Any]]]


def _entity_key(ent: dict[str, Any]) -> tuple:
    return (ent.get("start_char"), ent.get("end_char"), ent.get("label"), ent.get("text"))


def adjudicate_entities(model_records: Sequence[ModelAnnotation], threshold: float = 0.5) -> AdjudicationResult:
    if not model_records:
        return AdjudicationResult({}, [], {})
    total = len(model_records)
    base = dict(model_records[0].record)
    votes: dict[tuple, list[str]] = defaultdict(list)
    entity_by_key: dict[tuple, dict] = {}
    for item in model_records:
        for ent in item.record.get("entities") or []:
            key = _entity_key(ent)
            votes[key].append(item.model)
            entity_by_key.setdefault(key, dict(ent))
    consensus_entities = []
    disagreements = []
    for key, voters in sorted(votes.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1] or 0, str(kv[0][2]))):
        ratio = len(voters) / total
        ent = dict(entity_by_key[key])
        ent["vote_count"] = len(voters)
        ent["vote_ratio"] = round(ratio, 3)
        ent["voters"] = voters
        if ratio >= threshold:
            consensus_entities.append(ent)
        if ratio < 1.0:
            disagreements.append({"entity": ent, "voters": voters, "missing_from": [m.model for m in model_records if m.model not in voters]})
    base["entities"] = consensus_entities
    return AdjudicationResult(base, disagreements, {str(k): v for k, v in votes.items()})


def adjudicate_labels(predictions: dict[str, str], priority: Sequence[str] | None = None) -> dict[str, Any]:
    if not predictions:
        return {"label": None, "agreement_ratio": 0.0, "votes": {}}
    counts = Counter(predictions.values())
    top_count = max(counts.values())
    labels = [k for k, v in counts.items() if v == top_count]
    if len(labels) == 1:
        winner = labels[0]
    else:
        winner = labels[0]
        for model in priority or []:
            if predictions.get(model) in labels:
                winner = predictions[model]
                break
    return {"label": winner, "agreement_ratio": round(top_count / len(predictions), 3), "votes": dict(counts)}


def run_builtin_moe(text: str, model_names: Sequence[str] = ("spaCy+ruler",), resources_dir: str | None = None) -> AdjudicationResult:
    """Offline MoE for the app: currently runs deterministic annotators.

    The interface is intentionally model-agnostic so LLM/HF annotators can be added
    by wrapping their output as ``ModelAnnotation``.
    """
    records: list[ModelAnnotation] = []
    for name in model_names:
        nlp = load_spacy_model("en_core_web_sm", resources_dir=resources_dir, add_entity_ruler=("ruler" in name.lower()))
        ann = Annotator(nlp, resources_dir=resources_dir)
        records.append(ModelAnnotation(name, ann.annotate(text, include_text=True, include_verbs=True)))
    return adjudicate_entities(records)
