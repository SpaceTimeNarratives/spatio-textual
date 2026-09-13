# Changelog

## v0.4.0: unreleased

### Added
- Evidence-grounded spatial span and journey extraction with locally computed offsets.
- Rule, transformer and LLM-capable affect and journey components.
- Reusable reference validation, evaluation, review and provenance utilities.
- Public lexical-cue explanations for the rule sentiment and emotion analyzers.

### Changed
- Package schemas and evaluation-policy identifiers are project-independent.
- Python compatibility CI now covers Python 3.9, 3.11 and 3.12.
- Generated `spatio_textual.egg-info` metadata is no longer version-controlled.

### Fixed
- Editing a resolved place invalidates stale coordinates while retaining the
  original values in the human-review audit trail.
- Failed LLM affect and journey requests are recorded as backend errors rather
  than valid neutral or empty predictions.
- Affect evaluation refuses to score backend failures.
- Evidence-only journey references can match on grounded evidence spans.
- The Python 3.9 dependency path remains compatible with the tutorial spaCy
  model wheel.

### Migration notes
- Reference records now use `schema_version: spatio-textual-gold-0.1`.
- Affect output now uses `unsupported_emotion_labels` instead of the former
  conference-specific field name.
- Journey and affect evaluation policy identifiers now begin with
  `spatio-textual-`.

## v0.3.0: 2026-08-19

### Added
- Streamlit annotation app (app.py) with model selection, review queue, telemetry and exports (JSON / JSONL / CSV / CoNLL).
- Dockerfile and Hugging Face Space metadata (hf_space/README.md) for easy demo/deployment.
- Multi-model adjudication (MoE) workflow that produces consensus annotations and a disagreements review queue (spatio_textual/moe.py).
- Hugging Face transformer NER support alongside spaCy pipelines (spatio_textual/transformer_ner.py, model_registry.py).
- Sentiment & emotion modules with rule / HF / LLM-capable backends and distribution outputs (spatio_textual/sentiment.py, spatio_textual/emotion.py).
- Offline-safe place resolver/geocoder with ambiguity handling and fallbacks (spatio_textual/geocode.py).
- Lightweight LLM client for structured JSON classification (spatio_textual/llm.py).
- Telemetry helpers and token estimates for model calls (spatio_textual/telemetry.py).
- BIO/CoNLL utilities and converters (spatio_textual/formats.py).
- CLI refactor with new flags, worker support, HF & MoE options (spatio_textual/cli.py).
- Tutorial and tests (tutorials/full_day_end_to_end_tutorial.md, tests/test_core.py).

### Changed
- Version bumped to 0.3.0; pyproject.toml updated with clearer optional extras (app, transformers, llm) and dependency adjustments.
- README streamlined for v0.3 features and packaging metadata improved.
- Requirements split into requirements-lite/requirements-transformers/requirements-llm for easier installs.

### Breaking/migration notes
- Public API and CLI flags changed:
  - spatio_textual.__init__ exports were reorganised (e.g., SentimentAnalyzer, EmotionAnalyzer, run_builtin_moe, model registries). Update imports if you rely on old names.
  - CLI flags renamed/normalised (use `--ner-model`, `--sentiment-backend`, `--emotion-backend`, `--moe-models`, etc.). Update any scripts or CI that call the legacy CLI.
- Annotation record shape extended: records now include `segStartChar` / `segEndChar`, `telemetry`, `requires_review`, `review_notes`, and richer `event_data` / `verb_data`. Consumers should validate schema expectations when loading older exports.

### How to try
1. Quick local run (lightweight):
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
   python -m pip install -U pip wheel
   python -m pip install -r requirements-lite.txt
   streamlit run app.py
   ```
2. Docker: build the provided Dockerfile and run the container to serve the Streamlit app.
3. Tests: run `pytest` (workflow added at .github/workflows/tests.yml).

### Notes
- Transformer / LLM features require optional dependencies (see requirements-transformers.txt and requirements-llm.txt or install extras via `pip install -e '.[transformers]'` / `pip install -e '.[llm]'`).
- Telemetry is included by default; downstream export/ingest workflows should handle the new telemetry fields.

---
