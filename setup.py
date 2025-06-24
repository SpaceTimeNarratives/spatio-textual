# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name='spatio-textual',
    version='0.1.0',
    description="A library for spatial textual analysis with a focus on the Corpus of Lake District Writing and Holocaust Survivors' Testimonies.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ignatius Ezeani",
    author_email="igezeani@yahoo.com",
    url="https://github.com/SpaceTimeNarratives/spatio-textual",
    packages=find_packages(),
    install_requires=[
        'spacy>=3.0.0',
        'geonamescache',
        'tqdm'
    ],
    include_package_data=True,
    package_data={
        'spatio_textual': ['resources/*.txt']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
