Prediction history and telemetry

Every `/api/detect` call appends one entry to
`artifacts/prediction_history.json`.

Each entry records the decision, confidence, reason, per-model
predictions, processing time, and upload filename.

Telemetry endpoints read from that file:

- `GET /api/stats` returns total counts for `bird`, `drone`,
  `uncertain`, and `error` decisions, the average confidence across
  entries that carry a numeric confidence, and the 20 most recent
  detections.
- `GET /api/history.json` returns the full stored history.

The automated test suite backs up this file before running and restores
it afterwards so tests never pollute real history.
