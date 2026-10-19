Testing procedures

Run the suite with:

python -m pytest -q

Configuration lives in `pytest.ini`:

- `testpaths = tests`
- markers: `smoke` for real non-mocked model inference, registered for
  strict marker checking.

Useful selections:

- Fast unit tests only: `python -m pytest -m "not slow" -q`
- Real inference checks: `python -m pytest -m smoke -q`
- Single file: `python -m pytest tests/test_aeronet.py -q`

What the suite covers:

- Dataset discovery, validation, duplicate grouping, and annotation
  preview separation with synthetic images.
- Coordinator agreement, disagreement, single-model, and error paths.
- Metric calculation including unavailable-model handling.
- Upload validation and API routes.
- An image-only firewall test that fails if video, webcam, streaming,
  or legacy upload/download routes are registered.
- Artifact existence checks that require `scripts/train_models.py` to
  have been run first.

History safety:

The API test fixture backs up `artifacts/prediction_history.json`
before running and restores it afterwards, so tests never pollute real
telemetry.
