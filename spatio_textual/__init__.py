"""spatio-textual: spatial textual annotation for digital and spatial humanities."""
from __future__ import annotations

from .utils import Annotator, load_spacy_model, split_into_segments, save_annotations, load_annotations
from .sentiment import SentimentAnalyzer
from .emotion import EmotionAnalyzer
from .qa import Segment, segment_testimony, segment_testimony_file
from .analysis import analyze_records
from .formats import entities_to_bio, bio_to_entities, entities_to_conll, conll_to_entities

__all__ = [
    "Annotator",
    "load_spacy_model",
    "split_into_segments",
    "save_annotations",
    "load_annotations",
    "SentimentAnalyzer",
    "EmotionAnalyzer",
    "Segment",
    "segment_testimony",
    "segment_testimony_file",
    "analyze_records",
    "entities_to_bio",
    "bio_to_entities",
    "entities_to_conll",
    "conll_to_entities",
]

__version__ = "0.2.0"
