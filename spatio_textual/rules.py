from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Any, Iterable

import spacy

from .geocode import GeoResolver
from .telemetry import estimate_tokens
from .utils import DEFAULT_RESOURCES_DIR, RESOURCE_LABELS

# Deliberately small, transparent cue inventories for the teaching baseline.
# These are not claimed to be complete linguistic grammars.
SPATIAL_RELATION_TERMS = ("near", "beyond", "beside", "between", "within", "across")
DIRECTION_TERMS = ("to our left", "to our right", "to my left", "to my right", "north", "south", "east", "west")
MOVEMENT_SURFACE_FORMS = (
    "left", "leave", "travelled", "traveled", "travel", "moved", "move", "deported",
    "walked", "walk", "escaped", "escape", "fled", "flee", "returned", "return",
    "arrived", "arrive", "crossed", "cross", "hid", "hide",
)
TRANSPORT_PATTERNS = (
    r"\bby\s+(train|bus|car|lorry|truck|boat|ship|plane|air|foot|bicycle)\b",
    r"\bon\s+foot\b",
)
# Historical/travel prose often spells small quantities out (e.g. "six miles").
# Keeping this vocabulary explicit preserves the inspectability of the rule baseline.
NUMBER_EXPRESSION = (
    r"(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
)
DISTANCE_PATTERN = re.compile(
    rf"\b(?:(?:about|approximately|roughly|nearly|around)\s+)?"
    rf"{NUMBER_EXPRESSION}\s+"
    r"(?:miles?|kilometres?|kilometers?|km)"
    r"(?:\s+distant)?\b",
    flags=re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")


def _read_terms(path: Path) -> list[str]:
    if not path.exists():
        return []
    terms: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        term = raw.strip()
        if term and not term.startswith("#"):
            terms.append(term)
    return terms


def load_teaching_gazetteer(path: str | Path) -> list[dict[str, str]]:
    """Load a simple CSV gazetteer with at least ``text`` and ``label`` columns."""
    path = Path(path)
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        if not {"text", "label"}.issubset(fields):
            raise ValueError(f"Gazetteer {path} must contain text,label columns")
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if text and label:
                rows.append({k: (v or "") for k, v in row.items()})
    return rows


class RuleGazetteerAnnotator:
    """Transparent deterministic baseline for spatial annotation comparison.

    The baseline combines:

    - project resource lists (e.g. geo-nouns);
    - an optional bounded teaching gazetteer;
    - a small set of explicit regex/phrase rules for distance, time, movement,
      transport, direction and common spatial relations.

    It is intentionally inspectable and incomplete. It should not be presented
    as a state-of-the-art rule system or tuned on the final holdout set.
    """

    def __init__(
        self,
        *,
        resources_dir: str | Path | None = None,
        gazetteer_path: str | Path | None = None,
        include_project_resources: bool = True,
        case_sensitive: bool = False,
        link_places: bool = False,
        resolver: GeoResolver | None = None,
    ):
        self.resources_dir = Path(resources_dir) if resources_dir else DEFAULT_RESOURCES_DIR
        self.gazetteer_path = Path(gazetteer_path) if gazetteer_path else None
        self.include_project_resources = include_project_resources
        self.case_sensitive = case_sensitive
        self.link_places = link_places
        self.resolver = resolver or GeoResolver()

        self.nlp = spacy.blank("en")
        phrase_attr = "ORTH" if case_sensitive else "LOWER"
        self.ruler = self.nlp.add_pipe(
            "entity_ruler",
            config={"phrase_matcher_attr": phrase_attr, "overwrite_ents": False},
        )
        self._patterns: list[dict[str, Any]] = []
        self._build_patterns()

    def _build_patterns(self) -> None:
        patterns: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        if self.include_project_resources:
            for filename, label in RESOURCE_LABELS.items():
                for term in _read_terms(self.resources_dir / filename):
                    key = (label, term if self.case_sensitive else term.lower())
                    if key not in seen:
                        patterns.append({"label": label, "pattern": term})
                        seen.add(key)

        if self.gazetteer_path:
            for row in load_teaching_gazetteer(self.gazetteer_path):
                term = row["text"]
                label = row["label"]
                key = (label, term if self.case_sensitive else term.lower())
                if key not in seen:
                    patterns.append({"label": label, "pattern": term})
                    seen.add(key)

        for term in SPATIAL_RELATION_TERMS:
            key = ("SPATIAL_RELATION", term if self.case_sensitive else term.lower())
            if key not in seen:
                patterns.append({"label": "SPATIAL_RELATION", "pattern": term})
                seen.add(key)

        for term in DIRECTION_TERMS:
            key = ("DIRECTION", term if self.case_sensitive else term.lower())
            if key not in seen:
                patterns.append({"label": "DIRECTION", "pattern": term})
                seen.add(key)

        for term in MOVEMENT_SURFACE_FORMS:
            key = ("MOVEMENT_CUE", term if self.case_sensitive else term.lower())
            if key not in seen:
                patterns.append({"label": "MOVEMENT_CUE", "pattern": term})
                seen.add(key)

        if patterns:
            self.ruler.add_patterns(patterns)
        self._patterns = patterns

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    def annotate(self, text: str) -> dict[str, Any]:
        start_time = time.perf_counter()
        text = text or ""
        entities: list[dict[str, Any]] = []
        seen: set[tuple[int, int, str]] = set()

        doc = self.nlp(text)
        for ent in doc.ents:
            row = {
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "source": "rule:entity_ruler",
                "confidence": 1.0,
            }
            if self.link_places and ent.label_ in {"TOPONYM", "GPE", "LOC", "CITY", "COUNTRY", "CAMP", "PLACE"}:
                linked = self.resolver.resolve(ent.text, ent.label_, context=text)
                if linked:
                    row.update(linked)
            entities.append(row)
            seen.add((ent.start_char, ent.end_char, ent.label_))

        # Regex rules are added only when their exact span/label has not already
        # been produced by the phrase ruler.
        self._add_regex_matches(entities, seen, text, DISTANCE_PATTERN, "DISTANCE", "rule:distance_regex")
        self._add_regex_matches(entities, seen, text, YEAR_PATTERN, "TIME", "rule:year_regex")
        for pattern in TRANSPORT_PATTERNS:
            self._add_regex_matches(
                entities,
                seen,
                text,
                re.compile(pattern, flags=re.IGNORECASE),
                "TRANSPORT_CUE",
                "rule:transport_regex",
            )

        entities.sort(key=lambda row: (row["start_char"], row["end_char"], row["label"]))
        review = any(
            row.get("ambiguous") or row.get("resolution_status") == "unresolved"
            for row in entities
        )
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        return {
            "spans": entities,
            "requires_review": review,
            "review_notes": ["One or more linked places are ambiguous or unresolved."] if review else [],
            "telemetry": [{
                "task": "rule_gazetteer_spatial_annotation",
                "backend": "rules",
                "provider": "local",
                "model": "entity_ruler+regex",
                "latency_ms": latency_ms,
                "input_chars": len(text),
                "input_tokens_est": estimate_tokens(text),
                "output_tokens_est": estimate_tokens(str(entities)),
                "cost_usd_est": 0.0,
                "success": True,
                "error": None,
            }],
        }

    @staticmethod
    def _add_regex_matches(
        out: list[dict[str, Any]],
        seen: set[tuple[int, int, str]],
        text: str,
        pattern: re.Pattern[str],
        label: str,
        source: str,
    ) -> None:
        for match in pattern.finditer(text):
            key = (match.start(), match.end(), label)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "text": match.group(0),
                "label": label,
                "start_char": match.start(),
                "end_char": match.end(),
                "source": source,
                "confidence": 1.0,
            })


def filter_supported_gold_labels(
    spans: Iterable[dict[str, Any]],
    supported_labels: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter reference spans to the labels a rule baseline is intended to predict."""
    labels = set(supported_labels or {
        "TOPONYM", "GEONOUN", "SPATIAL_RELATION", "DISTANCE", "DIRECTION",
        "TIME", "MOVEMENT_CUE", "TRANSPORT_CUE",
    })
    return [dict(span) for span in spans if span.get("label") in labels]
