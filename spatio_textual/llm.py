from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import json, os, time

# ------------------ Shared helpers ------------------

def _tostr(x) -> str:
    return "" if x is None else str(x)

def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

# ---------- Sentiment helpers ----------

PROMPT_SENTIMENT = """\
You are a rater. Classify the SENTIMENT of the text as one of: positive, neutral, negative.
Return ONLY a single JSON object: {"label": "...", "score": 0.0-1.0}
Text:
"""

def _parse_sentiment_json(txt: str) -> Dict[str, Any]:
    try:
        obj = json.loads(txt.strip())
    except Exception:
        lo = txt.lower()
        lab = "positive" if "positive" in lo else "negative" if "negative" in lo else "neutral"
        return {"label": lab, "score": 0.5 if lab == "neutral" else 0.7}
    lab = _tostr(obj.get("label", "neutral")).strip().lower()
    if lab not in {"positive", "neutral", "negative"}:
        lab = "neutral"
    score = max(0.0, min(1.0, _safe_float(obj.get("score", 0.0))))
    return {"label": lab, "score": score}

# ---------- Emotion helpers ----------

EMOTION_LABELS = ["neutral", "joy", "surprise", "sadness", "fear", "anger", "disgust"]

PROMPT_EMOTION = """\
You are an emotion classifier. Classify the text into exactly one of:
neutral, joy, surprise, sadness, fear, anger, disgust.

Return ONLY strict JSON like:
{
  "label": "joy|surprise|sadness|fear|anger|disgust|neutral",
  "score": 0.0-1.0,
  "distribution": { "neutral": p, "joy": p, "surprise": p, "sadness": p, "fear": p, "anger": p, "disgust": p }  # optional
}
Text:
"""

def _parse_emotion_json(txt: str) -> Dict[str, Any]:
    try:
        obj = json.loads(txt.strip())
    except Exception:
        lo = txt.lower()
        lab = next((l for l in EMOTION_LABELS if l in lo), "neutral")
        return {"label": lab, "score": 0.5}
    lab = _tostr(obj.get("label", "neutral")).strip().lower()
    if lab not in EMOTION_LABELS:
        lab = "neutral"
    score = max(0.0, min(1.0, _safe_float(obj.get("score", 0.0))))
    dist = obj.get("distribution")
    if isinstance(dist, dict):
        s = sum(_safe_float(dist.get(k, 0.0)) for k in EMOTION_LABELS) or 1.0
        dist = {k: _safe_float(dist.get(k, 0.0)) / s for k in EMOTION_LABELS}
    else:
        dist = None
    out = {"label": lab, "score": score}
    if dist is not None:
        out["distribution"] = dist
    return out

# ------------------ Router ------------------

@dataclass
class LLMRouter:
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 128

    # ---- core chat once ----
    def _chat_once_json(self, user_prompt: str) -> str:
        """Send a single JSON-only chat to the configured provider and return the raw text."""
        p = (self.provider or "").lower()

        if p in {"openai", "openai_compat", "xai", "groq"}:
            try:
                from openai import OpenAI
            except Exception as e:
                raise RuntimeError("Install openai: pip install openai") from e
            # API key & base URL resolution (Groq/xAI often use OpenAI-compatible endpoints)
            api_key = self.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
            base_url = self.base_url or os.getenv("OPENAI_BASE_URL")
            if p == "groq":
                api_key = self.api_key or os.getenv("GROQ_API_KEY") or api_key
                base_url = self.base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            if p == "xai":
                api_key = self.api_key or os.getenv("OPENAI_API_KEY") or api_key
                base_url = self.base_url or os.getenv("XAI_BASE_URL", "https://api.x.ai")
            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            msgs = [
                {"role": "system", "content": "Return one JSON object only. No prose."},
                {"role": "user", "content": user_prompt},
            ]
            resp = client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""

        if p == "anthropic":
            try:
                from anthropic import Anthropic
            except Exception as e:
                raise RuntimeError("Install anthropic: pip install anthropic") from e
            client = Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
            msgs = [{"role": "user", "content": user_prompt}]
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="Return one JSON object only. No prose.",
                messages=msgs,
            )
            out = ""
            for block in resp.content or []:
                if getattr(block, "type", None) == "text":
                    out += getattr(block, "text", "")
                elif isinstance(block, dict) and block.get("type") == "text":
                    out += block.get("text", "")
            return out

        if p == "google":
            try:
                import google.generativeai as genai
            except Exception as e:
                raise RuntimeError("Install google-generativeai: pip install google-generativeai") from e
            genai.configure(api_key=self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel(self.model)
            resp = model.generate_content([{"role": "user", "parts": [{"text": user_prompt}]}])
            try:
                return resp.text or ""
            except Exception:
                return ""

        if p == "ollama":
            import requests
            url = self.base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
            r = requests.post(
                f"{url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "Return one JSON object only. No prose."},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=90,
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "message" in data and "content" in data["message"]:
                return data["message"]["content"]
            return json.dumps(data)

        raise ValueError(f"Unknown provider: {self.provider}")

    # ---- public tasks ----

    def sentiment(self, texts: Iterable[str], rate_limit_s: float = 0.0) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for t in texts:
            raw = self._chat_once_json(PROMPT_SENTIMENT + t)
            out.append(_parse_sentiment_json(raw))
            if rate_limit_s:
                time.sleep(rate_limit_s)
        return out

    def emotion(
        self,
        texts: Iterable[str],
        return_distribution: bool = True,
        rate_limit_s: float = 0.0
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for t in texts:
            raw = self._chat_once_json(PROMPT_EMOTION + t)
            parsed = _parse_emotion_json(raw)
            if not return_distribution:
                parsed.pop("distribution", None)
            out.append(parsed)
            if rate_limit_s:
                time.sleep(rate_limit_s)
        return out
