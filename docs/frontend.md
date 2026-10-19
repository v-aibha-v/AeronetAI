Frontend dashboard

`templates/index.html` is the single-page browser interface served by `GET /`. It uses plain HTML, CSS, and JavaScript with Three.js loaded from CDN (`cdnjs three.js r128`). There is no build step.

- Sidebar upload (`dropZone`, `fileInput`, `previewBox`, `btnAnalyze`): click or drag-and-drop one still image. Client validation mirrors `app.py` (`ALLOWED_EXTENSIONS`, `MAX_UPLOAD_BYTES`): extension check `/\.(jpg|jpeg|png)$/i` and 16 MiB limit before enabling Analyze.
- Model status (`modelStatus`): fetched from `GET /api/models`; shows Available / Unavailable / Error per agent via `statusBadge`.
- Analysis flow (`analyzeFile` is a top-level page function, later wrapped for scan styling): posts `FormData` field `image` to `POST /api/detect`, renders `decision`, `confidence`, `reason`, per-model cards (`modelCards`), and `uploaded_image` (`res-img`). Failures show in `msg-result`. `btnReset` clears the preview and resets the 3D mood.
- Evaluation table (`evalPanel`, `evalTableWrap`): fetched from `GET /api/evaluation`. Row order is `resnet18`, `mobilenetv3`, `efficientnetb0`, `logistic_regression_baseline`, `combined_coordinator` with Accuracy, Precision (drone), Recall (drone), Macro F1, Avg inference, Test samples, Coverage, Uncertain / failures. Best numeric column is highlighted via `Math.max`. HTTP 404 shows the train-first message.
- 3D layer: full-screen canvas (`bg3d`) with a procedural drone, bird, and star field; mood (`idle`, `scan`, `bird`, `drone`) follows upload and result state. If the CDN fails, the page logs a warning and runs without the background.
