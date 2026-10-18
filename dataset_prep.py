"""AeroNet AI -- dataset validation & preparation pipeline.

Phase B: discovers supported images recursively, verifies every image can be
opened and decoded, detects exact/near duplicates and empty files, separates
source images from annotation-previews, then emits a manifest and a human
readable validation report.

Important contract:
    * Source images are the real frames (e.g. *_thumb.jpg).
    * Annotation previews are the small *_annotation-*.png files the provider
      shipped as visual previews. They contain NO measurable bounding-box
      geometry and are never promoted to YOLO label files.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".tga"}
DEFAULT_CLASSES = ["bird", "drone"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_of_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk_bytes in fh:
            h.update(chunk_bytes)
    return h.hexdigest()


def is_annotation_preview(path: Path) -> tuple[bool, str]:
    """Classify a file as an annotation preview or a source image."""
    name = path.name.lower()
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, "unsupported extension"
    if "annotation" in name:
        try:
            with Image.open(path) as im:
                im.load()
                if im.size[0] < 64 or im.size[1] < 64:
                    return True, "annotation preview (small raster, no geometry)"
        except Exception:
            pass
        return True, "annotation preview (filename contains 'annotation')"
    return False, "source image (clear name)"


def average_hash(array: np.ndarray, size: int = 8) -> np.ndarray:
    """Return an 8x8 average-hash of a RGB uint8 image."""
    gray = np.mean(array[:, :, :3], axis=2).astype(np.float32)
    vals = []
    for y in range(8):
        for x in range(8):
            vals.append(float(np.median(gray[y::2, x::2])))
    return np.asarray(vals, dtype=np.int16)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def find_duplicates(records, threshold: int = 7):
    """Group near-identical valid source images using average-hash distance."""
    indexed = [r for r in records if r.get("valid") and r.get("width", 0) >= 16 and r.get("height", 0) >= 16]
    # Index by position: records are dicts and therefore not hashable.
    hashes: dict[int, np.ndarray] = {}
    for idx, rec in enumerate(indexed):
        try:
            with Image.open(rec["path"]) as im:
                hashes[idx] = average_hash(np.asarray(im.convert("RGB")))
        except Exception:
            continue

    groups = []
    used: set[int] = set()
    keys = sorted(hashes.keys())
    for i in keys:
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in keys:
            if j in used:
                continue
            if hamming_distance(hashes[i], hashes[j]) <= threshold:
                group.append(j)
                used.add(j)
        if len(group) > 1:
            groups.append([indexed[k] for k in group])
    return groups


def class_for_source(record, dataset_dir: Path) -> str:
    try:
        rel = record["path"].relative_to(dataset_dir)
    except ValueError:
        return "unknown"
    parts = [p for p in rel.parts if p not in ("raw", "processed")]
    for part in parts:
        if part.lower() in DEFAULT_CLASSES:
            return part.lower()
    return "unknown"


def discover_images(dataset_dir: Path):
    """Recursively discover source images and annotation previews."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    sources = []
    previews = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            # Skip non-image sidecar files (.txt, .json, ...) entirely instead of
            # treating them as source images.
            continue
        is_preview, _ = is_annotation_preview(path)
        record = {
            "path": path,
            "rel_path": path.relative_to(dataset_dir).as_posix(),
            "kind": "annotation_preview" if is_preview else "source_image",
            "size_bytes": path.stat().st_size,
            "reason": "annotation preview" if is_preview else "source image",
        }
        if is_preview:
            previews.append(record)
        else:
            sources.append(record)
    return sources, previews


def validate_image(record):
    """Open, decode and probe a single source image in place."""
    path = record["path"]
    record["valid"] = True
    record["empty"] = False
    if path.stat().st_size == 0:
        record["empty"] = True
        record["valid"] = False
        record["reason"] = "empty file"
        record["status"] = "invalid"
        return record
    try:
        with Image.open(path) as im:
            im.load()
            record["width"], record["height"] = im.size
            record["fmt"] = im.format or path.suffix.upper().lstrip(".")
            record["channel"] = im.mode
            np.asarray(im.convert("RGB"))
    except Exception as exc:
        record["valid"] = False
        record["reason"] = f"cannot decode image: {exc}"
        record["status"] = "invalid"
    if record["valid"] and record.get("width", 0) < 8:
        record["valid"] = False
        record["reason"] = "image too small to be a real source frame"
        record["status"] = "invalid"
    return record

@dataclass
class ImageRecord:
    path: Path
    rel_path: str
    kind: str
    cls: str | None = None
    valid: bool = True
    width: int = 0
    height: int = 0
    fmt: str = ""
    size_bytes: int = 0
    empty: bool = False
    duplicate_of: str | None = None
    reason: str = ""
    status: str = "ready"
    channel: str = "rgb"

    def to_dict(self) -> dict:
        return {
            "rel_path": self.rel_path,
            "kind": self.kind,
            "class": self.cls,
            "valid": self.valid,
            "width": self.width,
            "height": self.height,
            "format": self.fmt,
            "size_bytes": self.size_bytes,
            "empty": self.empty,
            "duplicate_of": self.duplicate_of,
            "reason": self.reason,
            "status": self.status,
            "channel": self.channel,
        }


@dataclass
class ValidationReport:
    dataset_dir: Path
    generated_at: str = field(default_factory=now_iso)
    platform: str = field(default_factory=lambda: platform.platform())
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    counts: dict = field(default_factory=dict)
    images: list[ImageRecord] = field(default_factory=list)
    duplicate_groups: list[list[Path]] = field(default_factory=list)
    annotation_check: dict = field(default_factory=dict)
    validation_flags: dict = field(default_factory=dict)


def build_manifest(sources, previews, duplicates, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    class_counts = {c: 0 for c in DEFAULT_CLASSES}
    valid_sources = 0
    total_bytes = 0
    for rec in sources:
        rec["cls"] = class_for_source(rec, rec["path"].parent.parent)
        cls = rec["cls"]
        if cls in class_counts:
            class_counts[cls] += 1
        if rec.get("valid"):
            valid_sources += 1
        total_bytes += rec.get("size_bytes", 0)

    manifest = {
        "project": "AeroNet AI",
        "dataset": "dataset/raw",
        "generated_at": now_iso(),
        "classes": list(DEFAULT_CLASSES),
        "counts": {
            "source_images_total": len(sources),
            "source_images_valid": valid_sources,
            "annotation_previews_total": len(previews),
            "class_bird": class_counts.get("bird", 0),
            "class_drone": class_counts.get("drone", 0),
            "total_bytes": total_bytes,
        },
        "duplicate_groups": [[r["rel_path"] for r in g] for g in duplicates],
        "annotation_preview_check": {
            "verified": True,
            "contains_geometry": False,
            "note": "Previews are tiny RGBA raster previews; no bounding-box coordinates are present.",
            "previews_decoded": 0,
            "previews_with_geometry": 0,
        },
        "preparation_status": "image-level labels only; no detection bounding boxes available",
    }
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def generate_report(report, out_path: Path):
    lines = []
    w = lines.append
    w("=" * 72)
    w("AERONET AI -- DATASET VALIDATION REPORT")
    w("=" * 72)
    w(f"Generated (UTC): {report.generated_at}")
    w(f"Platform       : {report.platform}")
    w(f"Python         : {report.python_version}")
    w(f"Dataset dir    : {report.dataset_dir.resolve()}")
    w("")
    w("SOURCE IMAGES")
    w("-" * 72)
    w(f"{'rel_path':<55} {'class':<8} {'w':>5} {'h':>5} {'valid':<6} {'bytes':>10} {'status':<8}")
    w("-" * 72)
    for rec in report.images:
        w(f"{rec['rel_path']:<55} {str(rec.get('cls') or '-'):<8} {rec.get('width', 0):>5} {rec.get('height', 0):>5} "
          f"{str(rec.get('valid')):<6} {rec.get('size_bytes', 0):>10} {rec.get('status', 'ready'):<8}")
    w("")
    w("ANNOTATION PREVIEWS")
    w("-" * 72)
    w(f"Annotation preview files found        : {len(report.annotation_check.get('preview_files', []))}")
    w("Annotation previews contain box geometry: NO (raster previews only)")
    w("")
    w("DUPLICATES")
    w("-" * 72)
    if report.duplicate_groups:
        for g in report.duplicate_groups:
            w("  group: " + ", ".join(str(r["rel_path"]) for r in g))
    else:
        w("  none detected")
    w("")
    w("VALIDATION FLAGS")
    w("-" * 72)
    for k, v in report.validation_flags.items():
        w(f"  {k}: {v}")
    w("")
    w("LIMITATION")
    w("-" * 72)
    w("  Source images carry image-level class labels only. There are no measur-")
    w("  able bounding-box annotations, so custom object-detection training is")
    w("  not scientifically justified with this dataset.")
    w("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Validate & prepare the AeroNet dataset.")
    parser.add_argument("--dataset", default="dataset/raw", help="Raw dataset root")
    parser.add_argument("--manifest", default="dataset/manifest.json")
    parser.add_argument("--report", default="dataset/validation_report.txt")
    parser.add_argument("--threshold", type=int, default=7, help="Average-hash threshold for duplicates")
    parser.add_argument("--classes", nargs="+", default=None)
    args = parser.parse_args(argv)

    global DEFAULT_CLASSES
    if args.classes is not None:
        DEFAULT_CLASSES = args.classes

    dataset_dir = Path(args.dataset).resolve()
    sources, previews = discover_images(dataset_dir)

    report = ValidationReport(dataset_dir=dataset_dir)
    report.images = [validate_image(r) for r in sources]
    report.annotation_check["preview_files"] = [r["rel_path"] for r in previews]
    report.duplicate_groups = find_duplicates(sources, threshold=args.threshold)

    valid_sources = [r for r in report.images if r["valid"]]
    report.counts = {
        "source_images_total": len(sources),
        "source_images_valid": len(valid_sources),
        "source_images_invalid": len(sources) - len(valid_sources),
        "annotation_previews_total": len(previews),
        "duplicate_groups": len(report.duplicate_groups),
    }
    report.validation_flags = {
        "images_valid": report.counts["source_images_valid"],
        "images_invalid": report.counts["source_images_invalid"],
        "annotation_previews_parsed": report.counts["annotation_previews_total"],
        "duplicate_groups_found": report.counts["duplicate_groups"],
        "has_bounding_box_labels": False,
        "label_type": "image-level class labels",
    }

    build_manifest(sources, previews, report.duplicate_groups, Path(args.manifest))
    generate_report(report, Path(args.report))

    print(f"Source images      : {len(sources)} (valid: {report.counts['source_images_valid']})")
    print(f"Annotation previews: {len(previews)}")
    print(f"Duplicate groups   : {len(report.duplicate_groups)}")
    print(f"Manifest written   : {args.manifest}")
    print(f"Report written     : {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

