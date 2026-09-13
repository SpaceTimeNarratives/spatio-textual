from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

# The package reference schema uses a task-oriented ontology. Off-the-shelf NER
# models use their own training ontologies (e.g. spaCy GPE/LOC/FAC or CoNLL LOC).
# Scoring raw label strings would therefore confuse ontology mismatch with span
# detection failure. These mappings are deliberately explicit and narrow.
MODEL_TO_GOLD_LABEL = {
    "GPE": "TOPONYM",
    "LOC": "TOPONYM",
    "LOCATION": "TOPONYM",
    "FAC": "TOPONYM",
    "CITY": "TOPONYM",
    "COUNTRY": "TOPONYM",
    "CONTINENT": "TOPONYM",
    "CAMP": "TOPONYM",
    "PLACE": "TOPONYM",
    "B-LOC": "TOPONYM",
    "I-LOC": "TOPONYM",
    "GEONOUN": "GEONOUN",
    "DATE": "TIME",
    "TIME": "TIME",
}

NER_NAMED_PLACE_GOLD_LABELS = {"TOPONYM"}
NER_HYBRID_GOLD_LABELS = {"TOPONYM", "GEONOUN"}


def harmonize_ner_entities(
    entities: Sequence[dict[str, Any]],
    *,
    include_temporal: bool = False,
    keep_unmapped: bool = False,
) -> list[dict[str, Any]]:
    """Map common model entity labels onto the package reference ontology.

    The original label is preserved in ``model_label`` and the original row is
    otherwise copied. Unmapped labels (PERSON/ORG/etc.) are dropped by default
    because they are outside the spatial NER task rather than false positives.
    """
    out: list[dict[str, Any]] = []
    for entity in entities:
        original = str(entity.get("label") or entity.get("entity_group") or entity.get("entity") or "")
        mapped = MODEL_TO_GOLD_LABEL.get(original)
        if mapped == "TIME" and not include_temporal:
            continue
        if mapped is None and not keep_unmapped:
            continue
        row = dict(entity)
        row["model_label"] = original
        row["label"] = mapped or original
        if "start" in row and "start_char" not in row:
            row["start_char"] = int(row["start"])
        if "end" in row and "end_char" not in row:
            row["end_char"] = int(row["end"])
        out.append(row)
    return out


def reference_spans_for_ner(
    record: dict[str, Any],
    *,
    include_geonouns: bool = False,
    include_temporal: bool = False,
) -> list[dict[str, Any]]:
    """Select the portion of the gold ontology a given NER comparison targets."""
    labels = set(NER_NAMED_PLACE_GOLD_LABELS)
    if include_geonouns:
        labels.add("GEONOUN")
    if include_temporal:
        labels.add("TIME")
    return [dict(span) for span in record.get("spans", []) if span.get("label") in labels]


def reference_spatial_reach(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all gold spans used to illustrate representational reach.

    This is **not** a fair NER accuracy denominator. It is used to show how much
    of the broader spatial-humanities ontology a named-entity model can express.
    """
    return [dict(span) for span in record.get("spans", [])]


def label_inventory(rows: Iterable[dict[str, Any]], key: str = "label") -> dict[str, int]:
    """Count labels for an inspectable ontology summary."""
    return dict(sorted(Counter(str(row.get(key) or "<none>") for row in rows).items()))


def supported_reference_fraction(
    record: dict[str, Any],
    supported_gold_labels: Iterable[str],
) -> dict[str, Any]:
    """Quantify ontology coverage before running a model.

    This answers a different question from recall: *even with perfect detection,
    what fraction of the reference span types could this output ontology encode?*
    """
    supported = set(supported_gold_labels)
    spans = list(record.get("spans", []))
    eligible = [span for span in spans if span.get("label") in supported]
    total = len(spans)
    return {
        "reference_total": total,
        "ontology_supported": len(eligible),
        "ontology_unsupported": total - len(eligible),
        "max_span_recall_from_ontology": round(len(eligible) / total, 6) if total else 1.0,
        "supported_labels": sorted(supported),
    }
