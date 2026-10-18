# Contributing to AeroNet AI

Thank you for your interest in contributing. This guide explains how to work on
AeroNet AI, which is an existing FastAPI + PyTorch image-classification project.

## Development environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the application locally:

```bash
python app.py
# Open http://127.0.0.1:8000/
```

## Running the tests

The project uses `pytest`. Run the full suite with:

```bash
python -m pytest -q
```

Markers:

- `smoke` — real, non-mocked model inference tests (may load model artifacts).
- `slow` — longer-running model artifact tests.

Run only the smoke tests:

```bash
python -m pytest -m smoke -q
```

## Documentation

Repository documentation lives in `README.md` and the `docs/` folder. If you add
or improve documentation:

1. Keep the change focused on one logical improvement.
2. Use a Conventional Commits message (for example `docs: ...`).
3. Update this file if the development workflow changes.
4. Ensure new files do not duplicate the README unnecessarily.

## Project structure

- `app.py` — FastAPI application and HTTP endpoints.
- `services/` — coordinator and evaluation logic.
- `models/` — model agent implementations.
- `artifacts/` — trained checkpoints and evaluation outputs.
- `dataset/` — dataset preparation, manifest, and validation report.
- `templates/` — browser frontend.
- `tests/` — automated tests.

See `README.md` for a full description of each directory.
