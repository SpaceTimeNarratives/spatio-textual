from __future__ import annotations

"""
annotate.py – user-friendly API that orchestrates utils.Annotator.

New:
- annotate_files: process one/many files with optional chunking; leverages other helpers.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Union

from .utils import Annotator, load_spacy_model, split_into_segments


# Lazy singleton initialiser kept for backward-compat
def _init_annotator():
    try:
        nlp = load_spacy_model('en_core_web_trf')
    except OSError:
        nlp = load_spacy_model('en_core_web_sm')
    return Annotator(nlp)

_annotator = None


def _get_annotator():
    global _annotator
    if _annotator is None:
        _annotator = _init_annotator()
    return _annotator


def annotate_text(text: str, *, include_entities: bool = True, include_verbs: bool = False) -> dict:
    return _get_annotator().annotate(text, include_entities=include_entities, include_verbs=include_verbs)


def annotate_texts(texts: List[str], file_id: Optional[str] = None, include_text: bool = False,
                   *, include_entities: bool = True, include_verbs: bool = False) -> List[dict]:
    return _get_annotator().annotate_texts(
        texts, file_id=file_id, start_seg_id=1, include_text=include_text,
        include_entities=include_entities, include_verbs=include_verbs
    )


def chunk_and_annotate_text(text: str, n_segments: int = 100, file_id: Optional[str] = None,
                            include_text: bool = False, *, include_entities: bool = True, include_verbs: bool = False) -> List[dict]:
    annotator = _get_annotator()
    segments = split_into_segments(text, n_segments=n_segments, nlp=annotator.nlp)
    return annotator.annotate_texts(
        segments, file_id=file_id, start_seg_id=1, include_text=include_text,
        include_entities=include_entities, include_verbs=include_verbs
    )


def chunk_and_annotate_file(path: str | Path, n_segments: int = 100, include_text: bool = False,
                            *, include_entities: bool = True, include_verbs: bool = False) -> List[dict]:
    return _get_annotator().annotate_file_chunked(
        path, n_segments=n_segments, include_text=include_text,
        include_entities=include_entities, include_verbs=include_verbs
    )


def annotate_files(inputs: Union[str, Path, Sequence[Union[str, Path]]],
                   *, glob_pattern: str = "*.txt", recursive: bool = True,
                   chunk: bool = True, n_segments: int = 100,
                   encoding: str = "utf-8", errors: str = "ignore",
                   include_text: bool = False,
                   include_entities: bool = True, include_verbs: bool = False) -> List[dict]:
    """
    Unified entry point to process one or many files.
    - chunk=True: sentence-safe chunking into ~n_segments per file
    - chunk=False: whole-file mode
    """
    if chunk:
        results: List[dict] = []
        annotator = _get_annotator()
        files = list(annotator._resolve_input_files(inputs, glob_pattern, recursive))
        for f in files:
            results.extend(annotator.annotate_file_chunked(
                f, n_segments=n_segments, encoding=encoding, errors=errors,
                include_text=include_text, include_entities=include_entities, include_verbs=include_verbs
            ))
        return results
    else:
        return _get_annotator().annotate_inputs(
            inputs, glob_pattern=glob_pattern, recursive=recursive,
            encoding=encoding, errors=errors, include_text=include_text,
            include_entities=include_entities, include_verbs=include_verbs
        )
