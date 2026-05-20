from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from .telemetry import estimate_tokens


class LLMClient:
    """Small structured-output adapter for common providers.

    The class is intentionally optional-dependency based. It is safe to import in
    lightweight tutorials, and provider SDKs are imported only when used.
    """

    def __init__(self, provider: str = "openai", model: str | None = None, api_key: str | None = None, base_url: str | None = None):
        self.provider = provider
        self.model = model or self.default_model(provider)
        self.api_key = api_key
        self.base_url = base_url

    @staticmethod
    def default_model(provider: str) -> str:
        return {
            "openai": "gpt-4.1-mini",
            "azure_openai": "gpt-4.1-mini",
            "anthropic": "claude-sonnet-4-5",
            "google_gemini": "gemini-3.5-flash",
            "groq": "llama-3.3-70b-versatile",
            "mistral": "mistral-large-latest",
            "huggingface_inference": "meta-llama/Llama-3.1-8B-Instruct",
            "ollama": "llama3.1",
        }.get(provider, "gpt-4.1-mini")

    def classify_json(self, task: str, text: str, labels: list[str], instructions: str) -> dict[str, Any]:
        prompt = (
            f"Task: {task}\n"
            f"Allowed labels: {labels}\n"
            f"Instructions: {instructions}\n"
            "Return strict JSON with keys: label, distribution, explanation. "
            "The distribution must contain all labels and sum approximately to 1.\n\n"
            f"Text:\n{text}"
        )
        start = time.perf_counter()
        out_text = ""
        error = None
        try:
            out_text = self._complete(prompt)
            data = self._extract_json(out_text)
        except Exception as exc:
            error = str(exc)
            data = {"label": "mixed", "distribution": {lab: 0.0 for lab in labels}, "explanation": error}
        data["telemetry"] = {
            "task": task,
            "backend": "llm",
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round((time.perf_counter() - start) * 1000, 3),
            "input_chars": len(text or ""),
            "input_tokens_est": estimate_tokens(prompt),
            "output_tokens_est": estimate_tokens(out_text),
            "cost_usd_est": None,
            "success": error is None,
            "error": error,
        }
        return data

    def _complete(self, prompt: str) -> str:
        if self.provider in {"openai", "azure_openai", "groq"}:
            from openai import AzureOpenAI, OpenAI
            if self.provider == "azure_openai":
                client = AzureOpenAI(
                    api_key=self.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=self.base_url or os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                )
            elif self.provider == "groq":
                client = OpenAI(api_key=self.api_key or os.getenv("GROQ_API_KEY"), base_url=self.base_url or "https://api.groq.com/openai/v1")
            else:
                client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"), base_url=self.base_url)
            res = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return res.choices[0].message.content or "{}"

        if self.provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
            res = client.messages.create(model=self.model, max_tokens=600, messages=[{"role": "user", "content": prompt}])
            return "".join(block.text for block in res.content if getattr(block, "type", "") == "text")

        if self.provider == "google_gemini":
            from google import genai
            client = genai.Client(api_key=self.api_key or os.getenv("GOOGLE_API_KEY"))
            res = client.models.generate_content(model=self.model, contents=prompt)
            return res.text or "{}"

        if self.provider == "mistral":
            from mistralai import Mistral
            client = Mistral(api_key=self.api_key or os.getenv("MISTRAL_API_KEY"))
            res = client.chat.complete(model=self.model, messages=[{"role": "user", "content": prompt}])
            return res.choices[0].message.content or "{}"

        if self.provider == "huggingface_inference":
            import httpx
            token = self.api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            url = self.base_url or f"https://api-inference.huggingface.co/models/{self.model}"
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            r = httpx.post(url, headers=headers, json={"inputs": prompt}, timeout=60)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return data[0].get("generated_text") or str(data)
            return str(data)

        if self.provider == "ollama":
            import httpx
            url = self.base_url or "http://localhost:11434/api/generate"
            r = httpx.post(url, json={"model": self.model, "prompt": prompt, "stream": False}, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "{}")

        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _extract_json(self, text: str) -> dict[str, Any]:
        text = (text or "{}").strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json\n", "", 1)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        return json.loads(text)
