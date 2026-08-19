from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

from .utils import Annotator, load_spacy_model, split_into_segments

_annotator: Annotator | None = None


def _get_annotator() -> Annotator:
    global _annotator
    if _annotator is None:
        _annotator = Annotator(load_spacy_model("en_core_web_sm"))
    return _annotator


def annotate_text(text: str, *, include_entities: bool = True, include_verbs: bool = False) -> dict:
    return _get_annotator().annotate(text, include_entities=include_entities, include_verbs=include_verbs, include_text=True)


def annotate_texts(texts: Sequence[str], file_id: Optional[str] = None, include_text: bool = False, *, include_entities: bool = True, include_verbs: bool = False) -> list[dict]:
    return _get_annotator().annotate_texts(texts, file_id=file_id, include_text=include_text, include_entities=include_entities, include_verbs=include_verbs)


def chunk_and_annotate_text(text: str, n_segments: Optional[int] = None, max_chars: int = 14000, file_id: Optional[str] = None, include_text: bool = True, *, include_entities: bool = True, include_verbs: bool = False) -> list[dict]:
    annotator = _get_annotator()
    segments = split_into_segments(text, n_segments=n_segments, nlp=annotator.nlp, max_chars=max_chars, as_records=True)
    return annotator.annotate_texts(segments, file_id=file_id, include_text=include_text, include_entities=include_entities, include_verbs=include_verbs)


def chunk_and_annotate_file(path: str | Path, n_segments: Optional[int] = None, max_chars: int = 14000, include_text: bool = True, *, include_entities: bool = True, include_verbs: bool = False) -> list[dict]:
    return _get_annotator().annotate_file_chunked(path, n_segments=n_segments, max_chars=max_chars, include_text=include_text, include_entities=include_entities, include_verbs=include_verbs)


def annotate_files(inputs: Union[str, Path, Sequence[Union[str, Path]]], glob_pattern: str = "*.txt", recursive: bool = True, chunk: bool = True, n_segments: Optional[int] = None, max_chars: int = 14000, include_text: bool = True, *, include_entities: bool = True, include_verbs: bool = False) -> list[dict]:
    ann = _get_annotator()
    if not chunk:
        return ann.annotate_inputs(inputs, glob_pattern=glob_pattern, recursive=recursive, include_text=include_text, include_entities=include_entities, include_verbs=include_verbs)
    records: list[dict] = []
    for f in ann._resolve_input_files(inputs, glob_pattern, recursive):
        records.extend(ann.annotate_file_chunked(f, n_segments=n_segments, max_chars=max_chars, include_text=include_text, include_entities=include_entities, include_verbs=include_verbs))
    return records
