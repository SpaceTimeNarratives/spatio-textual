from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "spatio_textual"
EXPECTED_RESOURCE_FILES = {
    "cleaned_holocaust_camps.txt",
    "combined_geonouns.txt",
    "family_terms.txt",
    "known_cities.txt",
    "non_verbals.txt",
}
LEGACY_WORKSHOP_PATHS = (
    ROOT / "spatio_textual_package_a_demo.ipynb",
    ROOT / "tutorials" / "full_day_end_to_end_tutorial.md",
    ROOT / "example-texts" / "long-text",
    ROOT / "example-texts" / "short-text",
)


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


def test_package_root_has_no_legacy_workshop_material():
    offenders = [str(path.relative_to(ROOT)) for path in LEGACY_WORKSHOP_PATHS if path.exists()]
    assert not offenders, "Workshop-only material remains in the package repository: " + ", ".join(offenders)


def test_package_resource_inventory_is_explicit():
    actual = {path.name for path in (PACKAGE / "resources").glob("*.txt")}
    assert actual == EXPECTED_RESOURCE_FILES
