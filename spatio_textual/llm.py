# spatio_textual/llm.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import os, json, time

# ---------- Small helpers ----------

LABELS = {"positive", "neutral", "negative"}

def _norm_label(s: str) -> str:
    s = (s or "").strip().lower()
    if s.startswith("pos"): return "positive"
    if s.startswith("neg"): return "negative"
    if s in LABELS: return s
    return "neutral"

def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

PROMPT_SENTIMENT = """\
You are a rater. Classify the SENTIMENT of the text as one of: positive, neutral, negative.
Return *only* JSON with keys: "label" (string) and "score" (float in [0,1]).
Text:
"""
# We’ll *strongly* nudge JSON-only outputs by wrapping with instructions.
SYSTEM_JSON_ONLY = (
    "You MUST answer with a single line of strict JSON. "
    "No prose, no backticks, no explanation."
)

def _build_messages(text: str, provider: str) -> Any:
    """
    Build a provider-appropriate chat payload.
    """
    # Default (OpenAI-style chat)
    msgs = [
        {"role": "system", "content": SYSTEM_JSON_ONLY},
        {"role": "user", "content": PROMPT_SENTIMENT + text},
    ]
    if provider in {"openai", "groq", "openai_compat", "xai"}:
        return msgs
    if provider == "anthropic":
        # Anthropic 'messages' expect 'content' per role
        return msgs
    if provider == "google":
        # Gemini expects a list of "parts"
        return [
            {"role": "user", "parts": [{"text": SYSTEM_JSON_ONLY + "\n" + PROMPT_SENTIMENT + text}]}
        ]
    if provider == "ollama":
        # We'll convert to a single prompt string
        return SYSTEM_JSON_ONLY + "\n" + PROMPT_SENTIMENT + text
    return msgs

def _parse_json_response(txt: str) -> Dict[str, Any]:
    """
    Parse model output into {"label","score"} safely.
    """
    # Try exact JSON first
    try:
        obj = json.loads(txt.strip())
        lab = _norm_label(obj.get("label", "neutral"))
        score = _safe_float(obj.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return {"label": lab, "score": score}
    except Exception:
        pass

    # Fallback: light extraction
    lo = txt.lower()
    lab = "positive" if "positive" in lo else "negative" if "negative" in lo else "neutral"
    # crude score hint
    score = 0.75 if lab != "neutral" else 0.5
    return {"label": lab, "score": score}

# ---------- Provider adapters ----------

@dataclass
class LLMRouter:
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 64

    def _client_openai(self):
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("Install openai: pip install openai") from e
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if self.provider in {"openai", "xai"} and not self.base_url:
            # openai official, or xAI which is OpenAI-compatible but needs base_url
            base_url = os.getenv("OPENAI_BASE_URL") if self.provider == "openai" else os.getenv("XAI_BASE_URL", "https://api.x.ai")
        else:
            base_url = self.base_url or os.getenv("OPENAI_BASE_URL")
        return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    def _client_anthropic(self):
        try:
            from anthropic import Anthropic
        except Exception as e:
            raise RuntimeError("Install anthropic: pip install anthropic") from e
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        return Anthropic(api_key=api_key)

    def _client_google(self):
        try:
            import google.generativeai as genai
        except Exception as e:
            raise RuntimeError("Install google-generativeai: pip install google-generativeai") from e
        api_key = self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        return genai

    def _client_groq(self):
        # Native Groq SDK (optional), else use OpenAI-compatible path with base_url.
        try:
            from groq import Groq
        except Exception:
            return None
        api_key = self.api_key or os.getenv("GROQ_API_KEY")
        return Groq(api_key=api_key)

    def _ollama_chat(self, prompt: str) -> str:
        # Minimal local Ollama chat call (no extra deps)
        import requests
        url = self.base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        resp = requests.post(
            f"{url}/api/chat",
            json={"model": self.model, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        # API streams chunks; final has "message"
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        # fallback if a different format
        return json.dumps(data)

    # ----- public tasks -----

    def sentiment(self, texts: Iterable[str], rate_limit_s: float = 0.0) -> List[Dict[str, Any]]:
        """
        Return a list of {"label","score"} for each text.
        """
        out: List[Dict[str, Any]] = []
        for t in texts:
            out.append(self._sent_one(t))
            if rate_limit_s > 0:
                time.sleep(rate_limit_s)
        return out

    # ----- per-provider single call -----

    def _sent_one(self, text: str) -> Dict[str, Any]:
        p = self.provider.lower()

        # OpenAI-style (official OpenAI, xAI via base_url, Together, Mistral's compat, etc.)
        if p in {"openai", "openai_compat", "xai"}:
            client = self._client_openai()
            msgs = _build_messages(text, provider=p)
            resp = client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},  # strong JSON nudge (OpenAI supports)
            )
            txt = resp.choices[0].message.content
            return _parse_json_response(txt)

        # Anthropic (Claude)
        if p == "anthropic":
            client = self._client_anthropic()
            msgs = _build_messages(text, provider=p)
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=msgs,
                system=SYSTEM_JSON_ONLY,
            )
            # content is a list of blocks
            txt = ""
            for block in (resp.content or []):
                if getattr(block, "type", None) == "text":
                    txt += getattr(block, "text", "")
                elif isinstance(block, dict) and block.get("type") == "text":
                    txt += block.get("text", "")
            return _parse_json_response(txt)

        # Google (Gemini)
        if p == "google":
            genai = self._client_google()
            model = genai.GenerativeModel(self.model)
            parts = _build_messages(text, provider=p)  # gemini-style parts
            resp = model.generate_content(parts)
            txt = ""
            try:
                txt = resp.text or ""
            except Exception:
                # manually join candidates/parts
                if hasattr(resp, "candidates"):
                    for c in resp.candidates or []:
                        try:
                            txt += c.content.parts[0].text
                        except Exception:
                            pass
            return _parse_json_response(txt)

        # Groq (native) OR OpenAI-compatible fallback
        if p == "groq":
            client = self._client_groq()
            msgs = _build_messages(text, provider="openai")  # same format
            if client is not None:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                txt = resp.choices[0].message.content
                return _parse_json_response(txt)
            # Fallback to OpenAI-compatible if GROQ SDK missing
            self.base_url = self.base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            self.api_key = self.api_key or os.getenv("GROQ_API_KEY")
            return self._sent_one(text)  # re-enter as openai_compat/xai path

        # Local Meta Llama via Ollama
        if p == "ollama":
            prompt = _build_messages(text, provider=p)  # returns combined string
            txt = self._ollama_chat(prompt)
            return _parse_json_response(txt)

        raise ValueError(f"Unknown provider: {self.provider}")
