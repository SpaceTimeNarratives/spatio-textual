# import spacy
# from .utils import Annotator, load_spacy_model


# # Initialize at module load
# def _init_annotator():
#     try:
#         # primary: transformer model
#         nlp = load_spacy_model('en_core_web_trf')
#     except OSError:
#         # fallback: small English model (usually installed by default)
#         nlp = load_spacy_model('en_core_web_sm')
#     return Annotator(nlp)

# # Instantiate and initialise the Annotator object
# _annotator = _init_annotator()

# def annotate_text(text: str) -> dict:
#     """
#     Annotate a single text string.
#     Usage:
#         from spatio_textual import annotate_text
#         result = annotate_text(\"Some text here.\")
#     """
#     return _annotator.annotate(text)

# annotator.py
from __future__ import annotations
import os
import threading
import spacy
from pathlib import Path
from .utils import Annotator, load_spacy_model

# ---- Config defaults (env overridable) ----
_DEFAULT_MODEL = os.getenv("SPACY_MODEL", "en_core_web_trf")
_FALLBACK_MODEL = os.getenv("SPACY_FALLBACK_MODEL", "en_core_web_sm")
_RESOURCES_DIR = os.getenv("SPATIAL_RESOURCES_DIR")  # optional override path
_ADD_ENTITY_RULER = os.getenv("SPACY_ADD_ENTITY_RULER", "1") != "0"
_AUTO_INIT = os.getenv("SPACY_AUTO_INIT", "1") != "0"  # set to 0 to skip eager init

# ---- Lazy singleton annotator ----
__ANNOTATOR_LOCK = threading.Lock()
__ANNOTATOR: Annotator | None = None
__MODEL_NAME: str | None = None
__RESOURCES_PATH: str | None = None

def _build_annotator(
    model_name: str | None = None,
    resources_dir: str | None = _RESOURCES_DIR,
    add_entity_ruler: bool = _ADD_ENTITY_RULER,
) -> Annotator:
    model = model_name or _DEFAULT_MODEL
    try:
        nlp = load_spacy_model(model, resources_dir=resources_dir, add_entity_ruler=add_entity_ruler)
    except OSError:
        # fallback if the transformer model isn't present
        nlp = load_spacy_model(_FALLBACK_MODEL, resources_dir=resources_dir, add_entity_ruler=add_entity_ruler)
    return Annotator(nlp, resources_dir=resources_dir)

def get_annotator() -> Annotator:
    """Return a process-local singleton Annotator, creating it on first use."""
    global __ANNOTATOR, __MODEL_NAME, __RESOURCES_PATH
    if __ANNOTATOR is None:
        with __ANNOTATOR_LOCK:
            if __ANNOTATOR is None:
                __ANNOTATOR = _build_annotator()
                __MODEL_NAME = _DEFAULT_MODEL
                __RESOURCES_PATH = _RESOURCES_DIR
    return __ANNOTATOR

def configure_annotator(
    model_name: str | None = None,
    resources_dir: str | None = None,
    add_entity_ruler: bool | None = None,
    force_reload: bool = True,
) -> None:
    """
    Optionally (re)configure the process-local annotator.
    Call this BEFORE annotate_text() if you want custom settings.
    """
    global __ANNOTATOR, __MODEL_NAME, __RESOURCES_PATH
    if not force_reload and __ANNOTATOR is not None:
        return
    with __ANNOTATOR_LOCK:
        __ANNOTATOR = _build_annotator(
            model_name=model_name or _DEFAULT_MODEL,
            resources_dir=resources_dir if resources_dir is not None else _RESOURCES_DIR,
            add_entity_ruler=_ADD_ENTITY_RULER if add_entity_ruler is None else add_entity_ruler,
        )
        __MODEL_NAME = model_name or _DEFAULT_MODEL
        __RESOURCES_PATH = resources_dir if resources_dir is not None else _RESOURCES_DIR

def annotate_text(text: str) -> dict:
    """
    Annotate a single text string.
    Usage:
        from spatio_textual.annotator import annotate_text
        result = annotate_text("Some text here.")
    """
    ann = get_annotator()
    return ann.annotate(text)

# Optional eager init (can be disabled by SPACY_AUTO_INIT=0)
if _AUTO_INIT:
    get_annotator()
