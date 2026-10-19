API endpoints

The application exposes an image-only HTTP surface.

- GET / returns the browser interface.
- POST /api/detect accepts one image in the multipart form field named
  image and returns the coordinator decision, confidence, reason,
  processing time, per-model results, and the uploaded image.
- GET /api/models reports per-agent availability and status.
- GET /api/evaluation returns the saved held-out evaluation report, or a
  404 message when it has not been computed yet.
- POST /api/evaluate reruns training and evaluation, then returns the
  new report.
- GET /api/manifest returns the dataset manifest.
- GET /api/stats returns prediction-history summaries.
- GET /api/history.json returns the stored prediction history.

Framework routes such as /docs, /redoc, and /openapi.json are registered
by FastAPI and are not part of the image-only application surface.
