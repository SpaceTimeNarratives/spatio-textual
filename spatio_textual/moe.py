from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from .model_registry import parse_ner_model
from .transformer_ner import HFNERAnnotator
from .utils import Annotator, load_spacy_model


@dataclass
class ModelAnnotation:
    model: str
    record: dict[str, Any]


@dataclass
class AdjudicationResult:
    consensus: dict[str, Any]
    disagreements: list[dict[str, Any]]
    votes: dict[str, list[str]]
    requires_review: bool = False


def _entity_key(ent: dict[str, Any], char_tolerance: int = 2) -> tuple:
    start = ent.get("start_char")
    end = ent.get("end_char")
    if isinstance(start, int):
        start = round(start / max(1, char_tolerance)) * char_tolerance
    if isinstance(end, int):
        end = round(end / max(1, char_tolerance)) * char_tolerance
    return (start, end, ent.get("label"), str(ent.get("text", "")).lower())


def adjudicate_entities(model_records: Sequence[ModelAnnotation], threshold: float = 0.5) -> AdjudicationResult:
    if not model_records:
        return AdjudicationResult({}, [], {}, False)
    total = len(model_records)
    base = dict(model_records[0].record)
    votes: dict[tuple, list[str]] = defaultdict(list)
    entity_by_key: dict[tuple, dict] = {}
    telemetry = []
    for item in model_records:
        telemetry.extend(item.record.get("telemetry") or [])
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
            disagreements.append({
                "entity": ent,
                "voters": voters,
                "missing_from": [m.model for m in model_records if m.model not in voters],
                "reason": "model_disagreement",
            })
    base["entities"] = consensus_entities
    base["telemetry"] = telemetry
    base["requires_review"] = bool(disagreements or any(e.get("ambiguous") or e.get("resolution_status") == "unresolved" for e in consensus_entities))
    base["review_notes"] = []
    if disagreements:
        base["review_notes"].append("MoE disagreement: human adjudication recommended before export.")
    return AdjudicationResult(base, disagreements, {str(k): v for k, v in votes.items()}, base["requires_review"])


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


def annotate_with_model(text: str, model_key: str, resources_dir: str | None = None, link_places: bool = True) -> ModelAnnotation:
    spec = parse_ner_model(model_key)
    if spec.backend == "hf":
        ann = HFNERAnnotator(spec.model, link_places=link_places)
        return ModelAnnotation(spec.key, ann.annotate(text, include_text=True))
    nlp = load_spacy_model(spec.model, resources_dir=resources_dir, add_entity_ruler=True)
    ann = Annotator(nlp, resources_dir=resources_dir, model_name=spec.model, link_places=link_places)
    return ModelAnnotation(spec.key, ann.annotate(text, include_text=True, include_verbs=True, include_events=True))


def run_builtin_moe(
    text: str,
    model_names: Sequence[str] = ("spacy:en_core_web_trf", "spacy:en_core_web_sm"),
    resources_dir: str | None = None,
    threshold: float = 0.5,
    link_places: bool = True,
) -> AdjudicationResult:
    records: list[ModelAnnotation] = []
    for name in model_names:
        records.append(annotate_with_model(text, name, resources_dir=resources_dir, link_places=link_places))
    return adjudicate_entities(records, threshold=threshold)
