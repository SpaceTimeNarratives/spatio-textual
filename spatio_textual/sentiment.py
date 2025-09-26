from __future__ import annotations
"""
sentiment.py — Pluggable sentiment analyzer with rule / hf / llm backends.

Backends
- "rule": lexicon-based polarity scorer (tiny fallback or Hu & Liu lists if present)
- "hf"  : HuggingFace Transformers pipeline (or LLMRouter if supplied)
- "llm" : Vendor-agnostic LLM via llm.py's LLMRouter or a user callable

Extras
- Optional signed score in [-1,1]: predict(..., include_signed=True)
- Robust lexicon loading from packaged resources/env
"""
from typing import Callable, List, Dict, Optional, Any
import math
import os
from pathlib import Path
from importlib import resources as _ires

# Optional router
try:
    from .llm import LLMRouter  # type: ignore
except Exception:  # pragma: no cover
    LLMRouter = None  # type: ignore

# ---------- Robust lexicon loading ----------
_DEF_POS = {
    "joy","happy","happiness","relief","love","peace","hope","safe","safely","freedom","free",
    "reunited","help","helped","support","protected","kind","kindness","welcomed","welcome"
}
_DEF_NEG = {
    "fear","afraid","terror","sad","sadness","cry","cried","anger","angry","hate","hated",
    "disgust","hunger","cold","death","dead","killed","beaten","sick","ill","hurt","pain",
    "lost","loss","lonely","alone","danger","unsafe","threat","starved","starvation"
}

def _read_wordlist(path: Path, *, encoding="latin-1", header_skip=35) -> set[str]:
    text = path.read_text(encoding=encoding, errors="ignore")
    lines = text.splitlines()
    start = header_skip if len(lines) > header_skip else 0
    items: set[str] = set()
    for raw in lines[start:]:
        s = raw.strip()
        if not s or s.startswith(";"):
            continue
        items.add(s.lower())
    return items

def _find_lexicon_paths() -> tuple[Optional[Path], Optional[Path]]:
    pos_env = os.getenv("POS_LEXICON")
    neg_env = os.getenv("NEG_LEXICON")
    if pos_env and neg_env:
        p1, p2 = Path(pos_env), Path(neg_env)
        if p1.exists() and p2.exists():
            return p1, p2

    base_env = os.getenv("SENTIMENT_LEXICON_DIR")
    if base_env:
        b = Path(base_env)
        p1, p2 = b / "positive-words.txt", b / "negative-words.txt"
        if p1.exists() and p2.exists():
            return p1, p2

    try:
        pkg = "spatio_textual.resources"
        pos_res = _ires.files(pkg).joinpath("positive-words.txt")
        neg_res = _ires.files(pkg).joinpath("negative-words.txt")
        if pos_res.is_file() and neg_res.is_file():
            with _ires.as_file(pos_res) as p1, _ires.as_file(neg_res) as p2:
                return Path(p1), Path(p2)
    except Exception:
        pass

    here = Path(__file__).parent
    p1, p2 = here / "resources" / "positive-words.txt", here / "resources" / "negative-words.txt"
    if p1.exists() and p2.exists():
        return p1, p2
    p1, p2 = here / "positive-words.txt", here / "negative-words.txt"
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
    POS_WORDS = _DEF_POS
    NEG_WORDS = _DEF_NEG

# ---------- Rule backend ----------

def _rule_score(text: str) -> Dict[str, float | str]:
    toks = [t.strip(".,;:!?\"'()[]{} ").lower() for t in text.split()]
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    raw = pos - neg
    score = math.tanh(raw / 3.0)  # in [-1,1]
    label = "neutral"
    if score > 0.15:
        label = "positive"
    elif score < -0.15:
        label = "negative"
    return {"label": label, "score": float(score)}

# ---------- Helpers ----------

def _norm_label(lab: str) -> str:
    lab = (lab or "").strip().lower()
    if lab.startswith("pos"): return "positive"
    if lab.startswith("neg"): return "negative"
    return lab if lab in {"positive","neutral","negative"} else "neutral"

_LABEL_MAP = {
    "positive":"pos","negative":"neg","neutral":"neu",
    "label_2":"pos","label_1":"neu","label_0":"neg",
}

def _signed_from_all_scores(all_scores: List[Dict[str, float]], normalize: bool = True) -> float:
    p_pos = p_neg = 0.0
    for d in all_scores:
        lab = _LABEL_MAP.get(str(d.get("label","")).lower(), str(d.get("label","")).lower())
        if lab in ("pos","positive"):
            p_pos = float(d.get("score",0.0))
        elif lab in ("neg","negative"):
            p_neg = float(d.get("score",0.0))
    if normalize:
        denom = max(p_pos + p_neg, 1e-9)
        return (p_pos - p_neg) / denom
    return p_pos - p_neg

def _signed_from_top(label: str, score: float) -> float:
    lab = (label or "").lower()
    if lab.startswith("pos"): return float(score)
    if lab.startswith("neg"): return -float(score)
    return 0.0

# ---------- Analyzer ----------

class SentimentAnalyzer:
    """
    backend: "rule" (default) | "hf" | "llm"
    model_name: HF model name for backend="hf"
    llm_fn: LLMRouter instance or callable(texts)->[{"label","score"}]
    """
    def __init__(self, backend: str = "rule", model_name: Optional[str] = None,
                 llm_fn: Optional[Callable[[List[str]], List[Dict]] | Any] = None):
        self.backend = backend
        self.model_name = model_name
        self.llm_fn = llm_fn
        self._pipe = None  # HF pipeline cache

    # HF
    def _ensure_hf_pipeline(self):
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline
        except Exception as e:
            raise RuntimeError("Transformers not installed. pip install transformers") from e
        model = self.model_name or "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self._pipe = pipeline("sentiment-analysis", model=model)

    # LLM
    def _ensure_llm_router(self):
        if LLMRouter is None:
            raise RuntimeError("LLMRouter not available. Ensure spatio_textual.llm is importable.")
        provider = os.getenv("LLM_PROVIDER")
        model = os.getenv("LLM_MODEL")
        if not provider or not model:
            raise ValueError("backend='llm' needs llm_fn or env LLM_PROVIDER + LLM_MODEL.")
        base_url = os.getenv("LLM_BASE_URL")
        return LLMRouter(provider=provider, model=model, base_url=base_url)

    def predict(self, texts: List[str], *, include_signed: bool = False,
                normalize_signed: bool = True) -> List[Dict]:
        # Rule
        if self.backend == "rule":
            out = [_rule_score(t) for t in texts]
            if include_signed:
                for r in out:
                    r["signed"] = float(r["score"])
            return out

        # LLM
        if self.backend == "llm":
            if LLMRouter is not None and isinstance(self.llm_fn, LLMRouter):
                preds = self.llm_fn.sentiment(texts)  # type: ignore[attr-defined]
            elif callable(self.llm_fn):
                preds = self.llm_fn(texts)  # type: ignore[call-arg]
            else:
                router = self._ensure_llm_router()
                preds = router.sentiment(texts)
            results = [{"label": _norm_label(p.get("label")), "score": float(p.get("score",0.0))} for p in preds]
            if include_signed:
                for r in results:
                    r["signed"] = _signed_from_top(r["label"], r["score"])
            return results

        # HF
        if self.backend == "hf":
            if LLMRouter is not None and isinstance(self.llm_fn, LLMRouter):
                preds = self.llm_fn.sentiment(texts)  # type: ignore[attr-defined]
                results = [{"label": _norm_label(p.get("label")), "score": float(p.get("score",0.0))} for p in preds]
                if include_signed:
                    for r in results:
                        r["signed"] = _signed_from_top(r["label"], r["score"])
                return results

            self._ensure_hf_pipeline()
            if include_signed:
                out = self._pipe(texts, return_all_scores=True)  # type: ignore[misc]
                results: List[Dict] = []
                for dist in out:
                    s_val = _signed_from_all_scores(dist, normalize=normalize_signed)
                    top = max(dist, key=lambda d: float(d.get("score",0.0)))
                    results.append({
                        "label": _norm_label(str(top.get("label","neutral"))),
                        "score": float(top.get("score",0.0)),
                        "signed": float(s_val),
                    })
                return results
            else:
                out = self._pipe(texts)  # type: ignore[misc]
                results = []
                for r in out:
                    lab = _norm_label(r.get("label","neutral"))
                    results.append({"label": lab, "score": float(r.get("score",0.0))})
                return results

        raise ValueError(f"Unknown backend: {self.backend}")
