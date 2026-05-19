from __future__ import annotations

import json
from typing import Any, Iterable, Sequence


def entities_to_bio(tokens: Sequence[str], entities: Sequence[dict[str, Any]], label_key: str = "label") -> list[str]:
    """Convert token-aligned entity spans into BIO tags.

    Entities should include ``start_token`` and ``end_token``. Character-only spans are
    ignored because token offsets are needed for a lossless BIO conversion.
    """
    tags = ["O"] * len(tokens)
    for ent in entities:
        start = ent.get("start_token")
        end = ent.get("end_token")
        label = ent.get(label_key) or ent.get("place_type") or "ENT"
        if start is None or end is None:
            continue
        try:
            start_i, end_i = int(start), int(end)
        except Exception:
            continue
        if start_i < 0 or end_i > len(tokens) or start_i >= end_i:
            continue
        tags[start_i] = f"B-{label}"
        for i in range(start_i + 1, end_i):
            tags[i] = f"I-{label}"
    return tags


def bio_to_entities(tokens: Sequence[str], tags: Sequence[str]) -> list[dict[str, Any]]:
    """Convert BIO tags to token-span entity dictionaries."""
    entities: list[dict[str, Any]] = []
    cur_label = None
    start = 0
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag == "O" or not tag:
            if cur_label is not None:
                entities.append({"text": " ".join(tokens[start:i]), "label": cur_label, "start_token": start, "end_token": i})
                cur_label = None
            continue
        prefix, _, label = tag.partition("-")
        if prefix == "B" or label != cur_label:
            if cur_label is not None:
                entities.append({"text": " ".join(tokens[start:i]), "label": cur_label, "start_token": start, "end_token": i})
            cur_label = label or tag
            start = i
    return entities


def entities_to_conll(tokens: Sequence[str], entities: Sequence[dict[str, Any]], doc_id: str = "doc") -> str:
    """Export tokens plus BIO tags as simple CoNLL text."""
    tags = entities_to_bio(tokens, entities)
    lines = [f"# doc_id = {doc_id}"]
    for tok, tag in zip(tokens, tags):
        lines.append(f"{tok}\t{tag}")
    return "\n".join(lines) + "\n"


def conll_to_entities(conll_text: str) -> list[dict[str, Any]]:
    """Parse a simple two-column CoNLL/BIO file into entity dictionaries."""
    tokens: list[str] = []
    tags: list[str] = []
    for line in conll_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        tokens.append(parts[0])
        tags.append(parts[-1])
    return bio_to_entities(tokens, tags)


def records_to_jsonl(records: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
