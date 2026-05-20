from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NERBackend = Literal["spacy", "hf"]


@dataclass(frozen=True)
class NERModelSpec:
    key: str
    backend: NERBackend
    model: str
    label: str
    description: str
    heavy: bool = False


NER_MODELS = {
    "spacy:en_core_web_trf": NERModelSpec(
        key="spacy:en_core_web_trf",
        backend="spacy",
        model="en_core_web_trf",
        label="spaCy transformer: en_core_web_trf",
        description="Recommended high-quality default when runtime allows.",
        heavy=True,
    ),
    "spacy:en_core_web_sm": NERModelSpec(
        key="spacy:en_core_web_sm",
        backend="spacy",
        model="en_core_web_sm",
        label="spaCy small: en_core_web_sm",
        description="Fast tutorial/dev model; lower recall than transformer models.",
    ),
    "hf:dslim/bert-base-NER": NERModelSpec(
        key="hf:dslim/bert-base-NER",
        backend="hf",
        model="dslim/bert-base-NER",
        label="HF BERT: dslim/bert-base-NER",
        description="BERT-base token classifier trained for PER/LOC/ORG/MISC.",
        heavy=True,
    ),
    "hf:dbmdz/bert-large-cased-finetuned-conll03-english": NERModelSpec(
        key="hf:dbmdz/bert-large-cased-finetuned-conll03-english",
        backend="hf",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        label="HF BERT-large: dbmdz CoNLL03 English",
        description="Larger BERT English NER model for comparison.",
        heavy=True,
    ),
}

DEFAULT_NER_MODEL = "spacy:en_core_web_trf"
TUTORIAL_NER_MODEL = "spacy:en_core_web_sm"

SENTIMENT_MODELS = {
    "none": "Do not run sentiment",
    "rule": "Offline rule baseline",
    "hf:cardiffnlp/twitter-roberta-base-sentiment-latest": "HF RoBERTa sentiment distribution",
    "hf:distilbert/distilbert-base-uncased-finetuned-sst-2-english": "HF DistilBERT binary sentiment baseline",
    "llm": "Provider LLM structured JSON sentiment",
}

EMOTION_MODELS = {
    "none": "Do not run emotion",
    "rule": "Offline Ekman-style rule baseline",
    "hf:j-hartmann/emotion-english-distilroberta-base": "HF DistilRoBERTa English emotion classifier",
    "llm": "Provider LLM structured JSON emotion",
}

LLM_PROVIDERS = [
    "openai",
    "azure_openai",
    "anthropic",
    "google_gemini",
    "groq",
    "mistral",
    "huggingface_inference",
    "ollama",
]


def parse_ner_model(key_or_model: str) -> NERModelSpec:
    if key_or_model in NER_MODELS:
        return NER_MODELS[key_or_model]
    if key_or_model.startswith("hf:"):
        model = key_or_model[3:]
        return NERModelSpec(key=key_or_model, backend="hf", model=model, label=f"HF: {model}", description="Custom Hugging Face token-classification model", heavy=True)
    if key_or_model.startswith("spacy:"):
        model = key_or_model.split(":", 1)[1]
        return NERModelSpec(key=key_or_model, backend="spacy", model=model, label=f"spaCy: {model}", description="Custom spaCy pipeline")
    # Backwards compatibility: raw spaCy model name
    return NERModelSpec(key=f"spacy:{key_or_model}", backend="spacy", model=key_or_model, label=f"spaCy: {key_or_model}", description="Custom spaCy pipeline")
