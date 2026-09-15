from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "spatio_textual"


def test_package_has_no_conference_layer_dependency():
    forbidden = (
        "projects.sh2026",
        "projects/sh2026",
        "tutorials/sh2026",
        "docs/sh2026",
        "spatial humanities 2026",
        "sh2026",
    )
    offenders: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        if any(marker in source for marker in forbidden):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, "Reusable package contains conference-specific coupling: " + ", ".join(offenders)
