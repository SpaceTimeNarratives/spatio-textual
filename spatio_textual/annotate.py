from __future__ import annotations

"""
annotate.py – lightweight module entry points that orchestrate utils.Annotator

Exposes:
- annotate_text(text) -> dict
- annotate_texts(list_of_texts, file_id=None) -> list[dict]
- chunk_and_annotate_text(text, n_segments=100, file_id=None) -> list[dict]
- chunk_and_annotate_file(path, n_segments=100, ...) -> list[dict]
"""

import spacy
from pathlib import Path
from typing import List, Optional

from .utils import Annotator, load_spacy_model, split_into_segments

# Initialize at module load (kept for backward-compat)
def _init_annotator():
    try:
        # primary: transformer model
        nlp = load_spacy_model('en_core_web_trf')
    except OSError:
        # fallback: small English model (usually installed by default)
        nlp = load_spacy_model('en_core_web_sm')
    return Annotator(nlp)

# Instantiate and initialise the Annotator object
_annotator = _init_annotator()

def annotate_text(text: str) -> dict:
    """
    Annotate a single text string.
    Usage:
        from spatio_textual import annotate_text
        result = annotate_text("Some text here.")
    """
    return _annotator.annotate(text)

def annotate_texts(texts: List[str], file_id: Optional[str] = None, include_text: bool = False) -> List[dict]:
    """
    Annotate a list of texts (e.g., pre-segmented chunks) with optional fileId and segId.
    """
    return _annotator.annotate_texts(texts, file_id=file_id, start_seg_id=1, include_text=include_text)

def chunk_and_annotate_text(text: str, n_segments: int = 100, file_id: Optional[str] = None, include_text: bool = False) -> List[dict]:
    """
    Chunk a single long text into ~n_segments (sentence-safe) and annotate each chunk.
    """
    segments = split_into_segments(text, n_segments=n_segments, nlp=_annotator.nlp)
    return _annotator.annotate_texts(segments, file_id=file_id, start_seg_id=1, include_text=include_text)

def chunk_and_annotate_file(path: str | Path, n_segments: int = 100, include_text: bool = False) -> List[dict]:
    """
    Read a file, split it into ~n_segments on sentence boundaries, and annotate each segment.
    """
    return _annotator.annotate_file_chunked(path, n_segments=n_segments, include_text=include_text)