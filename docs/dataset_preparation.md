Dataset preparation pipeline

`dataset_prep.py` validates and documents the bundled dataset without
changing application behavior.

- Discovers supported images recursively (`.jpg`, `.jpeg`, `.png`,
  `.webp`, `.bmp`, `.tif`, `.tiff`, `.tga`).
- Verifies every image opens and decodes; detects empty files and
  exact/near duplicates via average-hash distance.
- Separates real source images (for example `*_thumb.jpg`) from small
  `*_annotation-*.png` preview rasters, which carry no measurable
  bounding-box geometry and are never promoted to labels.
- Writes `dataset/manifest.json` (machine-readable inventory) and
  `dataset/validation_report.txt` (human-readable validation details).

Run with `python dataset_prep.py`; `--dataset`, `--manifest`,
`--report`, `--threshold`, and `--classes` override the defaults.
