Error handling

The API reports problems as JSON instead of crashing.

- Invalid upload (`/api/detect`): non-image, empty, oversized, or
  undecodable files return a JSON error describing the problem.
- No usable agent: when every agent is missing or fails, the coordinator
  returns an error decision with the reason
  `all_agents_unavailable_or_failed`.
- Missing evaluation (`GET /api/evaluation`): returns HTTP 404 with a
  message to run `POST /api/evaluate` or `python scripts/train_models.py`
  when no report exists yet.
- Failed retraining (`POST /api/evaluate`): returns HTTP 500 with the
  captured training stdout and stderr tails for diagnosis.
- Corrupt evaluation file: a missing or unreadable
  `artifacts/evaluation.json` is reported as unavailable, never replaced
  with invented numbers.
