#!/usr/bin/env bash
set -euo pipefail

# 1) Create & activate a venv in ./venv
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

# 2) Upgrade pip, install requirements and model
pip install --upgrade pip
pip install -r requirements.txt

# 3) Download the spaCy transformer model
python -m spacy download en-core-web-trf

echo
echo "✅ Virtual‑env created and dependencies installed."
echo "   Activate with: source venv/bin/activate"
