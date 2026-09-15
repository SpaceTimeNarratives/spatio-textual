from __future__ import annotations

import re
import time
from typing import Any

import spacy

from .telemetry import estimate_tokens

JOURNEY_FIELDS = ("start_location", "end_location", "transport_mode", "date", "journey_reason")

MOVEMENT_LEMMAS = {
    "arrive", "continue", "cross", "cycle", "depart", "drive", "escape", "flee", "fly",
    "go", "journey", "leave", "move", "reach", "relocate", "return", "ride", "sail", "settle",
    "take", "travel", "walk",
}
DESTINATION_LEMMAS = {"arrive", "reach", "relocate", "settle"}
SOURCE_LEMMAS = {"depart", "leave"}
IMPLIED_TRANSPORT = {
    "walk": "foot",
    "cycle": "bicycle",
    "drive": "car",
    "fly": "air",
    "sail": "ship",
}
TRANSPORT_ALIASES = {
    "rail": "train",
    "train": "train",
    "coach": "bus",
    "bus": "bus",
    "car": "car",
    "taxi": "car",
    "ferry": "ferry",
    "boat": "boat",
    "ship": "ship",
    "plane": "air",
    "air": "air",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "foot": "foot",
}

# Capitalised place-like phrases after a journey preposition are a transparent
# fallback when generic NER misses a place. Lower-case connectors support forms
# such as "Dar es Salaam" without introducing a task-specific gazetteer. A
# connector must be followed by another capitalised component, so phrases do not
# absorb ordinary continuation text such as "Inverness and checked ...".
PLACE_PHRASE = (
    r"[A-Z][\w'.-]*"
    r"(?:(?:\s+[A-Z][\w'.-]*)|(?:\s+(?:es|of|the|and)\s+[A-Z][\w'.-]*)){0,3}"
)
FROM_RE = re.compile(rf"\b(?i:from)\s+(?P<place>{PLACE_PHRASE})")
TO_RE = re.compile(rf"\b(?i:to|toward|towards|into)\s+(?P<place>{PLACE_PHRASE})")
ARRIVAL_RE = re.compile(rf"\b(?i:arrived|reached|settled|relocated)\s+(?i:in|at|to)\s+(?P<place>{PLACE_PHRASE})")
LEAVE_RE = re.compile(rf"\b(?i:left|departed(?:\s+from)?)\s+(?P<place>{PLACE_PHRASE})")
STATIC_IN_RE = re.compile(rf"\b(?i:in|at)\s+(?P<place>{PLACE_PHRASE})")

TRANSPORT_RE = re.compile(
    r"\b(?:by\s+|on\s+)(train|rail|coach|bus|car|taxi|ferry|boat|ship|plane|air|bicycle|bike|foot)\b",
    re.I,
)
DAY = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
# Specific surface-preserving patterns come before generic NER so the rule
# baseline keeps expressions such as "On Monday" or "In May" when that is what
# the source/reference records as the date phrase.
TIME_PATTERNS = [
    re.compile(rf"^On\s+(?:{DAY}|\d{{1,2}}\s+{MONTH})\b", re.I),
    re.compile(rf"^In\s+(?:{MONTH}|(?:19|20)\d{{2}})\b", re.I),
    re.compile(r"^(?:At sunrise|At dawn|Late that evening|The following morning|Two days later)\b", re.I),
    re.compile(r"\bbefore noon\b", re.I),
    re.compile(r"\b(?:19|20)\d{2}\b"),
    re.compile(rf"\b\d{{1,2}}\s+{MONTH}\b", re.I),
    re.compile(rf"\b{DAY}\b", re.I),
    re.compile(rf"\b{MONTH}\b", re.I),
]
PURPOSE_TO_RE = re.compile(r"\bto\s+(visit|attend|deliver|meet|work|study|join|escape)\b[^,.;]*", re.I)
FOR_REASON_RE = re.compile(r"\bfor\s+(?:an?\s+|the\s+)?[^,.;]+", re.I)

NEGATIVE_PATTERNS = [
    re.compile(r"\bnever\b", re.I),
    re.compile(r"\b(?:did\s+not|didn't|was\s+not|wasn't)\b", re.I),
    re.compile(r"\bplanned\s+to\b.*\b(?:cancelled|canceled)\b", re.I),
    re.compile(r"\bintended\s+to\b.*\b(?:cancelled|canceled)\b", re.I),
]


def _normalise_space(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value.strip(" \t\n\r,.;:"))
    return cleaned or None


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return _normalise_space(match.group("place")) if match else None


class RuleDependencyJourneyExtractor:
    """Transparent non-generative journey baseline.

    The extractor combines generic spaCy syntax/NER with small, documented
    movement/preposition rules. It is intentionally bounded: missing information
    remains null, contextual inheritance is explicit, and no geocoder or LLM is
    consulted. The formal v1 implementation is developed on the separate
    a development corpus rather than a frozen evaluation holdout.
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        try:
            self.nlp = spacy.load(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"RuleDependencyJourneyExtractor requires spaCy model {model_name!r}; no blank-model fallback is allowed."
            ) from exc

    @staticmethod
    def _negative(sentence: str) -> bool:
        return any(pattern.search(sentence) for pattern in NEGATIVE_PATTERNS)

    @staticmethod
    def _movement_lemmas(sent) -> list[str]:
        lemmas: list[str] = []
        for token in sent:
            lemma = (token.lemma_ or token.text).lower()
            if lemma in MOVEMENT_LEMMAS:
                lemmas.append(lemma)
        return lemmas

    @staticmethod
    def _places(sent) -> list[str]:
        out: list[str] = []
        for ent in sent.ents:
            if ent.label_ in {"GPE", "LOC", "FAC"} and ent.text not in out:
                out.append(ent.text)
        for token in sent:
            if token.pos_ == "PROPN" and token.dep_ == "pobj" and token.head.lower_ in {"from", "to", "toward", "towards", "in", "at", "into"}:
                phrase = _normalise_space(" ".join(t.text for t in token.subtree))
                if phrase and phrase not in out:
                    out.append(phrase)
        return out

    @staticmethod
    def _fallback_place(pattern: re.Pattern[str], sentence: str) -> str | None:
        return _first_match(pattern, sentence)

    def _source_destination(self, sent, movement: list[str]) -> tuple[str | None, str | None]:
        sentence = sent.text
        source = self._fallback_place(FROM_RE, sentence)
        destination = self._fallback_place(TO_RE, sentence)
        places = self._places(sent)

        if source is None and any(lemma in SOURCE_LEMMAS for lemma in movement):
            source = self._fallback_place(LEAVE_RE, sentence)
            if source is None and places:
                source = places[0]

        if destination is None and any(lemma in DESTINATION_LEMMAS for lemma in movement):
            destination = self._fallback_place(ARRIVAL_RE, sentence)
            if destination is None and places:
                destination = places[-1]

        if source is None:
            for token in sent:
                if token.lower_ == "from":
                    pobj = next((child for child in token.children if child.dep_ == "pobj"), None)
                    if pobj is not None:
                        source = _normalise_space(" ".join(t.text for t in pobj.subtree))
                        break
        if destination is None:
            for token in sent:
                if token.lower_ in {"to", "toward", "towards", "into"}:
                    pobj = next((child for child in token.children if child.dep_ == "pobj"), None)
                    if pobj is not None:
                        destination = _normalise_space(" ".join(t.text for t in pobj.subtree))
                        break
        return source, destination

    @staticmethod
    def _transport(sentence: str, movement: list[str]) -> tuple[str | None, str]:
        match = TRANSPORT_RE.search(sentence)
        if match:
            return TRANSPORT_ALIASES.get(match.group(1).lower(), match.group(1).lower()), "explicit"
        lower = sentence.lower()
        if "took the train" in lower or "took a train" in lower:
            return "train", "explicit"
        if "took the bus" in lower or "took a bus" in lower:
            return "bus", "explicit"
        if "took the ferry" in lower or "took a ferry" in lower:
            return "ferry", "explicit"
        for lemma in movement:
            if lemma in IMPLIED_TRANSPORT:
                return IMPLIED_TRANSPORT[lemma], "contextual_inference"
        return None, "missing"

    @staticmethod
    def _date(sent) -> str | None:
        for pattern in TIME_PATTERNS:
            match = pattern.search(sent.text)
            if match:
                return _normalise_space(match.group(0))
        for ent in sent.ents:
            if ent.label_ in {"DATE", "TIME"}:
                return _normalise_space(ent.text)
        return None

    @staticmethod
    def _reason(sentence: str) -> tuple[str | None, str]:
        match = FOR_REASON_RE.search(sentence)
        if match:
            value = _normalise_space(match.group(0))
            if value and not re.search(r"\b(?:days?|weeks?|months?|years?|hours?|minutes?)\b", value, re.I):
                return value, "explicit"
        match = PURPOSE_TO_RE.search(sentence)
        if match:
            return _normalise_space(match.group(0)), "explicit"
        if re.search(r"\bnew contract\b", sentence, re.I):
            return "new contract", "contextual_inference"
        return None, "missing"

    def _previous_place(self, previous_sent) -> str | None:
        if previous_sent is None:
            return None
        places = self._places(previous_sent)
        if places:
            return places[-1]
        return self._fallback_place(STATIC_IN_RE, previous_sent.text)

    @staticmethod
    def _needs_origin_inheritance(sentence: str, movement: list[str], source: str | None, destination: str | None) -> bool:
        if source is not None or destination is None:
            return False
        if re.search(r"\bfrom\s+(?:there|here)\b", sentence, re.I):
            return True
        if any(lemma in {"relocate", "settle", "return", "continue"} for lemma in movement):
            return True
        return False

    @staticmethod
    def _status(value: str | None, explicit: bool = True) -> str:
        if value is None:
            return "missing"
        return "explicit" if explicit else "contextual_inference"

    def extract(self, text: str, *, file_id: str = "document") -> dict[str, Any]:
        started = time.perf_counter()
        doc = self.nlp(text or "")
        sentences = list(doc.sents)
        journeys: list[dict[str, Any]] = []

        for index, sent in enumerate(sentences):
            sentence = sent.text.strip()
            if not sentence or self._negative(sentence):
                continue
            movement = self._movement_lemmas(sent)
            if not movement:
                continue

            source, destination = self._source_destination(sent, movement)
            previous = sentences[index - 1] if index > 0 else None
            source_inferred = False
            if self._needs_origin_inheritance(sentence, movement, source, destination):
                inherited = self._previous_place(previous)
                if inherited:
                    source = inherited
                    source_inferred = True

            if source is None and destination is None:
                continue

            transport, transport_status = self._transport(sentence, movement)
            date = self._date(sent)
            reason, reason_status = self._reason(sentence)

            use_previous = source_inferred and previous is not None
            if use_previous:
                evidence_start = previous.start_char
                evidence_end = sent.end_char
            else:
                evidence_start = sent.start_char
                evidence_end = sent.end_char
            evidence = doc.text[evidence_start:evidence_end]

            statuses = {
                "start_location": self._status(source, explicit=not source_inferred),
                "end_location": self._status(destination),
                "transport_mode": transport_status,
                "date": self._status(date),
                "journey_reason": reason_status,
            }
            requires_review = any(value == "contextual_inference" for value in statuses.values())
            journeys.append({
                "journeyId": f"{file_id}_rule_j{len(journeys) + 1}",
                "fileId": file_id,
                "start_location": source,
                "end_location": destination,
                "transport_mode": transport,
                "date": date,
                "journey_reason": reason,
                "evidence_quote": evidence,
                "evidence_start_char": evidence_start,
                "evidence_end_char": evidence_end,
                "evidence_grounded": True,
                "explicit_or_inferred": statuses,
                "requires_review": requires_review,
                "review_notes": ["Contains bounded contextual inference."] if requires_review else [],
                "source": "rule_dependency_v1",
            })

        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        return {
            "journeys": journeys,
            "requires_review": any(journey["requires_review"] for journey in journeys),
            "review_notes": ["At least one journey contains contextual inference."] if any(journey["requires_review"] for journey in journeys) else [],
            "telemetry": [{
                "task": "journey_extraction",
                "backend": "rules+dependency",
                "provider": "local",
                "model": f"rule_dependency_v1+{self.model_name}",
                "latency_ms": latency_ms,
                "input_chars": len(text or ""),
                "input_tokens_est": estimate_tokens(text or ""),
                "output_tokens_est": estimate_tokens(str(journeys)),
                "cost_usd_est": 0.0,
                "success": True,
                "error": None,
            }],
        }
