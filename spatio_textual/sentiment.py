from __future__ import annotations
"""
sentiment.py — Pluggable sentiment analyzer with robust lexicon loading.

Backends:
- "rule": lexicon-based polarity scorer (now loading Hu & Liu lists or fallbacks)
- "hf"  : HuggingFace Transformers pipeline (or LLMRouter if provided)
- "llm" : Vendor-agnostic LLM via `llm.py`'s LLMRouter or a user callable

Extras:
- Optional signed score in [-1, 1]: `predict(..., include_signed=True)`

Lexicon loading priority (first match wins):
1) Explicit paths via env: POS_LEXICON, NEG_LEXICON
2) A provided resources directory (set ENV `SENTIMENT_LEXICON_DIR`)
3) Package resources: `spatio_textual/resources/positive-words.txt` etc
4) Module-adjacent fallback: `sentiment.py`'s sibling files
5) Tiny built-in fallback set (keeps the package working if files are missing)

To include the Hu & Liu lists in a package build, ensure they are in
`spatio_textual/resources/` and add package-data in your build config, e.g.:

pyproject.toml (setuptools):
[tool.setuptools.package-data]
"spatio_textual.resources" = [
  "positive-words.txt",
  "negative-words.txt"
]

Or MANIFEST.in:
recursive-include spatio_textual/resources *.txt
"""
from typing import Callable, List, Dict, Optional, Any
import math
import os
from pathlib import Path

# Optional hook (only if available)
try:
    from .llm import LLMRouter  # your llm.py
except Exception:  # pragma: no cover
    LLMRouter = None  # type: ignore

# -----------------------
# Robust lexicon loading
# -----------------------
from importlib import resources as _ires

_DEF_POS = {
    "joy", "happy", "happiness", "relief", "love", "peace", "hope", "safe", "safely", "freedom", "free",
    "reunited", "help", "helped", "support", "protected", "kind", "kindness", "welcomed", "welcome"
}
_DEF_NEG = {
    "fear", "afraid", "terror", "sad", "sadness", "cry", "cried", "anger", "angry", "hate", "hated",
    "disgust", "hunger", "cold", "death", "dead", "killed", "beaten", "sick", "ill", "hurt", "pain",
    "lost", "loss", "lonely", "alone", "danger", "unsafe", "threat", "starved", "starvation"
}


def _read_wordlist(path: Path, *, encoding: str = "latin-1", header_skip: int = 35) -> set[str]:
    """Read a lexicon file, skipping header lines and comments.
    Hu & Liu lists have ~35 line header; we also strip blanks and ';' comments.
    """
    text = path.read_text(encoding=encoding, errors="ignore")
    lines = text.splitlines()
    # If header_skip is too big for a compact file, cap it
    start = header_skip if len(lines) > header_skip else 0
    items: set[str] = set()
    for raw in lines[start:]:
        s = raw.strip()
        if not s or s.startswith(";"):
            continue
        items.add(s.lower())
    return items


def _find_lexicon_paths() -> tuple[Optional[Path], Optional[Path]]:
    """Find POS/NEG lexicon files using the priority described in the module docstring."""
    # 1) Explicit env paths
    pos_env = os.getenv("POS_LEXICON")
    neg_env = os.getenv("NEG_LEXICON")
    if pos_env and neg_env:
        p1, p2 = Path(pos_env), Path(neg_env)
        if p1.exists() and p2.exists():
            return p1, p2

    # 2) Resources dir env
    base_env = os.getenv("SENTIMENT_LEXICON_DIR")
    if base_env:
        b = Path(base_env)
        p1 = b / "positive-words.txt"
        p2 = b / "negative-words.txt"
        if p1.exists() and p2.exists():
            return p1, p2

    # 3) Package resources
    try:
        pkg = "spatio_textual.resources"
        pos_res = _ires.files(pkg).joinpath("positive-words.txt")
        neg_res = _ires.files(pkg).joinpath("negative-words.txt")
        if pos_res.is_file() and neg_res.is_file():
            # Convert Traversable to real file path by extracting to temp, or open() directly
            with _ires.as_file(pos_res) as p1, _ires.as_file(neg_res) as p2:
                return Path(p1), Path(p2)
    except Exception:
        pass

    # 4) Module-adjacent (e.g., when running from repo without packaging)
    here = Path(__file__).parent
    p1 = here / "resources" / "positive-words.txt"
    p2 = here / "resources" / "negative-words.txt"
    if p1.exists() and p2.exists():
        return p1, p2

    # Could also try directly next to file (no resources/ subdir)
    p1 = here / "positive-words.txt"
    p2 = here / "negative-words.txt"
    if p1.exists() and p2.exists():
        return p1, p2

    return None, None


try:
    _pos_path, _neg_path = _find_lexicon_paths()
    if _pos_path and _neg_path:
        POS_WORDS = _read_wordlist(_pos_path)
        NEG_WORDS = _read_wordlist(_neg_path)
    else:
        POS_WORDS = _DEF_POS
        NEG_WORDS = _DEF_NEG
except Exception:
    # Always keep package importable
    POS_WORDS = _DEF_POS
    NEG_WORDS = _DEF_NEG


# ---------------- rule backend ----------------

def _rule_score(text: str) -> Dict[str, float | str]:
    toks = [t.strip(".,;:!?\"'()[]{} ").lower() for t in text.split()]
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    raw = pos - neg
    # squashed score in [-1,1]
    score = math.tanh(raw / 3.0)
    label = "neutral"
    if score > 0.15:
        label = "positive"
    elif score < -0.15:
        label = "negative"
    return {"label": label, "score": float(score)}


# ---------------- helpers ----------------

def _norm_label(lab: str) -> str:
    lab = (lab or "").strip().lower()
    if lab.startswith("pos"):
        return "positive"
    if lab.startswith("neg"):
        return "negative"
    if lab in {"positive", "neutral", "negative"}:
        return lab
    return "neutral"

# Some HF models use explicit labels; others (e.g., cardiffnlp) use LABEL_0/1/2
_LABEL_MAP = {
    "positive": "pos", "negative": "neg", "neutral": "neu",
    "label_2": "pos", "label_1": "neu", "label_0": "neg",
}

def _signed_from_all_scores(all_scores: List[Dict[str, float]], normalize: bool = True) -> float:
    """Map a pipeline(return_all_scores=True) output (list of {label,score}) to s in [-1,1]."""
    p_pos = p_neg = 0.0
    for d in all_scores:
        lab = _LABEL_MAP.get(str(d.get("label", "")).lower(), str(d.get("label", "")).lower())
        if lab in ("pos", "positive"):
            p_pos = float(d.get("score", 0.0))
        elif lab in ("neg", "negative"):
            p_neg = float(d.get("score", 0.0))
    if normalize:
        denom = max(p_pos + p_neg, 1e-9)
        return (p_pos - p_neg) / denom
    return p_pos - p_neg


def _signed_from_top(label: str, score: float) -> float:
    lab = (label or "").lower()
    if lab.startswith("pos"):
        return float(score)
    if lab.startswith("neg"):
        return -float(score)
    return 0.0


class SentimentAnalyzer:
    """
    Parameters
    ----------
    backend : str
        "rule" (default), "hf", or "llm".
    model_name : Optional[str]
        HF model name for backend="hf" (e.g., "cardiffnlp/twitter-roberta-base-sentiment-latest").
        Ignored for "rule". For "llm" you set model via env or llm_fn router.
    llm_fn : Optional[Callable | LLMRouter]
        - If backend="llm":
            * If llm_fn is an LLMRouter, we call llm_fn.sentiment(texts).
            * If llm_fn is a callable, we call llm_fn(texts) and expect [{"label","score"}, ...].
            * If llm_fn is None, we autoconfigure an LLMRouter from env vars.
        - If backend="hf":
            * If llm_fn is an LLMRouter, we also use that path (handy single code path).
            * Else we lazily create a transformers pipeline.
    """
    def __init__(self, backend: str = "rule", model_name: Optional[str] = None,
                 llm_fn: Optional[Callable[[List[str]], List[Dict]] | Any] = None):
        self.backend = backend
        self.model_name = model_name
        self.llm_fn = llm_fn
        self._pipe = None  # HF pipeline cache

    # ------------- HF helpers -------------
    def _ensure_hf_pipeline(self):
        """Lazily create a transformers sentiment pipeline."""
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline  # lazy import
        except Exception as e:
            raise RuntimeError(
                "Hugging Face Transformers not installed. "
                "Install with: pip install transformers"
            ) from e
        model = self.model_name or "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self._pipe = pipeline("sentiment-analysis", model=model)

    # ------------- LLM helpers -------------
    def _ensure_llm_router(self):
        """If llm_fn is None, try to build an LLMRouter from environment variables."""
        if LLMRouter is None:
            raise RuntimeError(
                "LLMRouter not available. Ensure spatio_textual.llm is importable."
            )
        provider = os.getenv("LLM_PROVIDER")
        model = os.getenv("LLM_MODEL")
        if not provider or not model:
            raise ValueError(
                "backend='llm' requires either llm_fn or environment variables "
                "LLM_PROVIDER and LLM_MODEL (e.g., LLM_PROVIDER=openai, LLM_MODEL=gpt-4o-mini)."
            )
        base_url = os.getenv("LLM_BASE_URL")  # for openai_compat/xai/together etc.
        return LLMRouter(provider=provider, model=model, base_url=base_url)

    # ------------- API -------------
    def predict(self, texts: List[str], *, include_signed: bool = False, normalize_signed: bool = True) -> List[Dict]:
        """
        Return a list of dicts for each text.
        Default keys: {"label","score"}
        If include_signed=True, also adds {"signed"} in [-1,1].
        """
        if self.backend == "rule":
            out = [_rule_score(t) for t in texts]
            if include_signed:
                # rule "score" is already in [-1,1]; mirror it to "signed"
                for r in out:
                    r["signed"] = float(r["score"])  # identical scale
            return out

        # --- LLM path (router or callable) ---
        if self.backend == "llm":
            # 1) Router instance supplied
            if LLMRouter is not None and isinstance(self.llm_fn, LLMRouter):
                preds = self.llm_fn.sentiment(texts)  # type: ignore[attr-defined]
            # 2) User callable supplied
            elif callable(self.llm_fn):
                preds = self.llm_fn(texts)  # type: ignore[call-arg]
            else:
                # 3) Autoconfigure router from env
                router = self._ensure_llm_router()
                preds = router.sentiment(texts)
            results = [{"label": _norm_label(p.get("label")), "score": float(p.get("score", 0.0))} for p in preds]
            if include_signed:
                for r in results:
                    # signed from top-1 label/score
                    r["signed"] = _signed_from_top(r["label"], r["score"]) if isinstance(r["label"], str) else 0.0
            return results

        # --- HF path ---
        if self.backend == "hf":
            # If caller passed an LLMRouter, allow it here too (single path)
            if LLMRouter is not None and isinstance(self.llm_fn, LLMRouter):
                preds = self.llm_fn.sentiment(texts)  # type: ignore[attr-defined]
                results = [{"label": _norm_label(p.get("label")), "score": float(p.get("score", 0.0))} for p in preds]
                if include_signed:
                    for r in results:
                        r["signed"] = _signed_from_top(r["label"], r["score"]) if isinstance(r["label"], str) else 0.0
                return results

            # Otherwise use transformers pipeline
            self._ensure_hf_pipeline()
            if include_signed:
                # Need per-class probabilities to build signed → request full distribution
                out = self._pipe(texts, return_all_scores=True)  # type: ignore[misc]
                results: List[Dict] = []
                for dist in out:
                    s_val = _signed_from_all_scores(dist, normalize=normalize_signed)
                    top = max(dist, key=lambda d: float(d.get("score", 0.0)))
                    results.append({
                        "label": _norm_label(str(top.get("label", "neutral"))),
                        "score": float(top.get("score", 0.0)),
                        "signed": float(s_val),
                    })
                return results
            else:
                out = self._pipe(texts)  # type: ignore[misc]
                results = []
                for r in out:
                    lab = _norm_label(r.get("label", "neutral"))
                    results.append({"label": lab, "score": float(r.get("score", 0.0))})
                return results

        raise ValueError(f"Unknown backend: {self.backend}")