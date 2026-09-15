from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional


@dataclass
class Telemetry:
    task: str
    backend: str
    model: str
    provider: Optional[str] = None
    latency_ms: float = 0.0
    input_chars: int = 0
    input_tokens_est: int = 0
    output_tokens_est: int = 0
    cost_usd_est: Optional[float] = None
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_tokens(text: str | None) -> int:
    """Cheap, dependency-free token estimate for telemetry.

    For precise billing, provider SDK tokenizers can be plugged in later. This
    estimate is intentionally conservative and works offline for tutorials.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def merge_telemetry(record: dict[str, Any], telemetry: Telemetry | dict[str, Any]) -> dict[str, Any]:
    item = telemetry.to_dict() if isinstance(telemetry, Telemetry) else dict(telemetry)
    record.setdefault("telemetry", [])
    record["telemetry"].append(item)
    return record


def timed(task: str, backend: str, model: str, provider: str | None = None):
    """Decorator returning ``(result, telemetry)`` from a no-arg callable."""

    def decorator(fn: Callable[[], Any]):
        def wrapper(input_text: str = ""):
            start = time.perf_counter()
            tel = Telemetry(
                task=task,
                backend=backend,
                model=model,
                provider=provider,
                input_chars=len(input_text or ""),
                input_tokens_est=estimate_tokens(input_text),
            )
            try:
                result = fn()
                tel.output_tokens_est = estimate_tokens(str(result))
                return result, tel.to_dict()
            except Exception as exc:  # pragma: no cover - caller usually catches
                tel.success = False
                tel.error = str(exc)
                return None, tel.to_dict()
            finally:
                tel.latency_ms = round((time.perf_counter() - start) * 1000, 3)

        return wrapper

    return decorator
