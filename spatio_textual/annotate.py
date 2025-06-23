import spacy
from .utils import Annotator, load_spacy_model

# Initialize at module load
_nlp = load_spacy_model()
_annotator = Annotator(_nlp)

def annotate_text(text: str) -> dict:
    """
    Annotate a single text string.
    Usage:
        from spatio_textual import annotate_text
        result = annotate_text(\"Some text here.\")
    """
    return _annotator.annotate(text)