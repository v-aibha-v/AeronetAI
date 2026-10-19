Upload limits

Only still images are accepted.

- Allowed extensions: `.jpg`, `.jpeg`, and `.png`.
- Maximum upload size: 16 MiB (`MAX_UPLOAD_BYTES = 16 * 1024 * 1024` in
  `app.py`).
- Empty uploads are rejected.
- Files that cannot be opened and decoded as images are rejected.
- Uploads are processed in memory and from a temporary file that is
  removed after the prediction completes; uploads are not kept on disk.
