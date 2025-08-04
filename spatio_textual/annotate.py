import spacy
from .utils import Annotator, load_spacy_model


# Initialize at module load
def _init_annotator():
    try:
        # primary: transformer model
        nlp = load_spacy_model('en_core_web_trf')
    except OSError:
        # fallback: small English model (usually installed by default)
        nlp = load_spacy_model('en_core_web_sm')
    return Annotator(nlp)

# _nlp = load_spacy_model()
_annotator = _init_annotator()

def annotate_text(text: str) -> dict:
    """
    Annotate a single text string.
    Usage:
        from spatio_textual import annotate_text
        result = annotate_text(\"Some text here.\")
    """
    return _annotator.annotate(text)