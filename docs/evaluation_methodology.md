Evaluation methodology

`services/evaluation_service.py` evaluates every available agent plus the
legacy baseline and the coordinator on one shared held-out split.

- Fixed seed `42`; stratified split with ratios train 0.70, val 0.15,
  test 0.15.
- Near-duplicates are grouped by same-class difference hash
  (distance <= 3) so copies land in the same split.
- Test images are never used for fitting or tuning; preprocessing comes
  from artifacts only.
- Positive class for binary precision/recall is `drone`.
- Per-class and macro-averaged metrics accompany binary metrics so weak
  performance on one class is not hidden.
- The report is written to `artifacts/evaluation.json` and served by
  `GET /api/evaluation`. Missing artifacts are listed as unavailable,
  never replaced with invented numbers.
