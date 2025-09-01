from __future__ import annotations
import os
import threading
import spacy
import json
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


# --- Health check ---
def annotator_ready() -> bool:
    """Return True if the process-local annotator is already constructed."""
    return " __ANNOTATOR" in globals() and (globals().get("__ANNOTATOR") is not None)

def annotator_info(init: bool = True) -> dict:
    """
    If init=True (default), ensures the annotator is constructed (lazy init)
    and returns full details. If init=False, returns a light snapshot without
    loading the model; 'ready' tells you if it's already built.
    """
    if not init and not annotator_ready():
        return {
            "ready": False,
            "auto_init": _AUTO_INIT,
            "configured_model": _DEFAULT_MODEL,
            "fallback_model": _FALLBACK_MODEL,
            "resources_dir": _RESOURCES_DIR or "",
        }

    ann = get_annotator()  # may lazily build
    nlp = ann.nlp
    meta = getattr(nlp, "meta", {}) or {}
    return {
        "ready": True,
        "spacy_version": spacy.__version__,
        "lang": getattr(nlp, "lang", None),
        "model_name": meta.get("name") or meta.get("pipeline"),
        "model_meta_version": meta.get("version"),
        "pipes": list(nlp.pipe_names),
        "has_entity_ruler": "entity_ruler" in nlp.pipe_names,
        "resources_dir": str(getattr(ann, "resources_dir", "")),
        "resource_counts": {
            "geonouns": len(getattr(ann, "geonouns", [])),
            "camps": len(getattr(ann, "camps", [])),
            "ambiguous_cities": len(getattr(ann, "ambiguous_cities", [])),
            "non_verbals": len(getattr(ann, "non_verbal", [])),
            "family_terms": len(getattr(ann, "family", [])),
        },
    }

def print_annotator_info(pretty: bool = True, init: bool = True) -> None:
    info = annotator_info(init=init)
    import json
    print(json.dumps(info, ensure_ascii=False, indent=2 if pretty else None))