"""Compatibility entry point for legacy ``setup.py`` tooling.

All project metadata lives in ``pyproject.toml`` so build frontends cannot
publish conflicting versions, dependencies, Python requirements or licences.
"""

from setuptools import setup


setup()
