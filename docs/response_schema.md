Detect response schema

`POST /api/detect` returns JSON with these top-level fields:

- `decision`: the coordinator decision (`bird`, `drone`, `uncertain`,
  or an error outcome).
- `confidence`: averaged class score for the decided class, or null when
  the agents disagree or no agent works.
- `reason`: short machine-readable decision reason such as
  `available_models_agree`, `single_model_available`,
  `models_disagree`, or `all_agents_unavailable_or_failed`.
- `processing_time_ms`: total request time.
- `models`: per-agent results for `resnet18`, `mobilenetv3`, and
  `efficientnetb0`.
- `result`: the full coordinator result object.
- `uploaded_image`: the uploaded image as a data URI for display.
- `file_name`: sanitized upload filename.

Each per-agent result contains `model_name`, `predicted_class`,
`class_scores`, `probabilities`, `inference_time_ms`, `status`, and
`error`.
