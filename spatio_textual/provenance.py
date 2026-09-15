from __future__ import annotations

import hashlib
import platform
import sys
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

SECRET_MARKERS = ("api_key", "apikey", "token", "secret", "password", "credential")


def sha256_text(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def redact_secret_like_keys(value: Any) -> Any:
    """Return a deep redacted copy of nested config-like data.

    Keys whose lowercase name contains a common secret marker are retained so
    the configuration shape remains inspectable, but their values are replaced
    with ``<redacted>``. The input object is not modified.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_text = str(key)
            if any(marker in key_text.lower() for marker in SECRET_MARKERS):
                out[key] = "<redacted>"
            else:
                out[key] = redact_secret_like_keys(item)
        return out
    if isinstance(value, list):
        return [redact_secret_like_keys(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secret_like_keys(item) for item in value)
    return deepcopy(value)


def build_run_manifest(
    *,
    input_text: str | None = None,
    config: dict[str, Any] | None = None,
    git_commit: str | None = None,
    package_name: str = "spatio-textual",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a secret-safe reproducibility manifest for tutorial/demo exports."""
    try:
        package_version = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        package_version = None

    return {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "package": package_name,
        "package_version": package_version,
        "git_commit": git_commit,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "input_sha256": sha256_text(input_text),
        "input_chars": len(input_text) if input_text is not None else None,
        "config": redact_secret_like_keys(config or {}),
    }
