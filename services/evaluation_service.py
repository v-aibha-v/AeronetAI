"""AeroNet AI -- held-out evaluation service.

Evaluates ResNet-18, MobileNetV3, and EfficientNet-B0 plus the legacy
LogisticRegression baseline on the same held-out test split, and evaluates the
decision coordinator on that same split.

Design rules (enforced by code):
* Fixed random seed; stratified, near-duplicate-grouped train/validation/test split.
* Test data is never used for fitting or tuning.
* Preprocessing comes from the model artifacts only (no test-dependent tuning).
* Per-class and macro-averaged metrics are reported alongside binary metrics
  (positive class: ``drone``) so weak performance on one class is not hidden.
* Results are saved to ``artifacts/evaluation.json``; nothing is fabricated.
* If artifacts are missing, evaluation reports them as unavailable instead of
  inventing numbers.

The dataset is very small and imbalanced (5 bird / 55 drone source images),
so metrics are indicative of the pipeline, not a statistically sound claim of
real-world performance.  This limitation is recorded in the report itself.
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `python services/evaluation_service.py`

DATASET_DIR = ROOT / "dataset" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"
RESNET18_ARTIFACT = ARTIFACTS_DIR / "resnet18_bird_drone.pt"
MOBILENET_ARTIFACT = ARTIFACTS_DIR / "mobilenet_v3_bird_drone.pt"
EFFICIENTNET_ARTIFACT = ARTIFACTS_DIR / "efficientnet_b0_bird_drone.pt"
LR_ARTIFACT = ARTIFACTS_DIR / "drone_bird_feature_lr.pkl"
EVALUATION_ARTIFACT = ARTIFACTS_DIR / "evaluation.json"

CLASS_NAMES = ["bird", "drone"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
POSITIVE_CLASS = "drone"  # documented positive class for binary precision/recall
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RESNET18_IMAGE_SIZE = 160
MOBILENET_IMAGE_SIZE = 224
LR_IMAGE_SIZE = 160
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Source images carry names like X_thumb.jpg; annotation previews are the small
# *_annotation-*.png rasters and are never used as classification samples.
SOURCE_NAME_MARKER = "_thumb"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _ImageListDataset(Dataset):
    def __init__(self, image_paths, labels, transform) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), int(self.labels[idx])


def list_source_images(dataset_dir: Path):
    """Discover decodable source images under class subdirectories.

    Annotation previews (names containing ``annotation``) are excluded: they
    are tiny rasters with no measurable labels and are not classification
    samples.  Returns (paths, labels, class_names, invalid).
    """
    class_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    class_names = [p.name for p in class_dirs]
    paths: list[Path] = []
    labels: list[int] = []
    invalid: list[tuple[str, str]] = []

    for label_idx, class_dir in enumerate(class_dirs):
        for image_path in sorted(p for p in class_dir.iterdir() if p.is_file()):
            if "annotation" in image_path.name.lower():
                continue
            try:
                with Image.open(image_path) as im:
                    im.load()
            except Exception as exc:
                invalid.append((str(image_path.relative_to(dataset_dir)), str(exc)))
                continue
            paths.append(image_path)
            labels.append(label_idx)

    return paths, labels, class_names, invalid


def stratified_split(paths, labels, train_ratio, val_ratio, test_ratio, seed):
    """Make reproducible stratified splits without separating near-duplicate frames."""
    hashes = []
    for path in paths:
        with Image.open(path) as image:
            pixels = np.asarray(
                image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            )
        hashes.append(pixels[:, 1:] > pixels[:, :-1])

    parents = list(range(len(paths)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left in range(len(paths)):
        for right in range(left + 1, len(paths)):
            if int(labels[left]) != int(labels[right]):
                continue
            distance = int(np.count_nonzero(hashes[left] != hashes[right]))
            if distance <= 3:
                left_root = find(left)
                right_root = find(right)
                if left_root != right_root:
                    parents[right_root] = left_root

    groups_by_root: dict[int, list[int]] = {}
    for index in range(len(paths)):
        groups_by_root.setdefault(find(index), []).append(index)
    groups_by_class: dict[int, list[list[int]]] = {}
    for group in groups_by_root.values():
        groups_by_class.setdefault(int(labels[group[0]]), []).append(group)

    rng = random.Random(seed)
    train_paths, val_paths, test_paths = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for label, class_groups in groups_by_class.items():
        rng.shuffle(class_groups)
        n_groups = len(class_groups)
        if n_groups >= 3:
            n_test = max(1, int(round(n_groups * test_ratio)))
            n_val = max(1, int(round(n_groups * val_ratio)))
            while n_test + n_val >= n_groups:
                if n_val > 1:
                    n_val -= 1
                else:
                    n_test -= 1
            n_train = n_groups - n_val - n_test
        elif n_groups == 2:
            n_train, n_val, n_test = 1, 0, 1
        else:
            n_train, n_val, n_test = n_groups, 0, 0

        split_groups = (
            class_groups[:n_train],
            class_groups[n_train:n_train + n_val],
            class_groups[n_train + n_val:n_train + n_val + n_test],
        )
        split_outputs = (
            (train_paths, train_labels),
            (val_paths, val_labels),
            (test_paths, test_labels),
        )
        for class_split_groups, (split_paths, split_labels) in zip(
            split_groups, split_outputs
        ):
            for group in class_split_groups:
                split_paths.extend(paths[index] for index in group)
                split_labels.extend([label] * len(group))

    return (
        train_paths, val_paths, test_paths,
        np.array(train_labels), np.array(val_labels), np.array(test_labels),
    )


def class_counts(labels, class_names):
    return {class_names[idx]: int((labels == idx).sum()) for idx in range(len(class_names))}


def _eval_transform(image_size: int, augment: bool = False):
    ops = []
    if augment:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def _unavailable_metrics(n_total: int, y_true: list[str] | None = None) -> dict[str, Any]:
    """Metrics block for a model that could not produce predictions."""
    supports = {
        name: sum(label == name for label in y_true) if y_true is not None else 0
        for name in CLASS_NAMES
    }
    per_class = {}
    for name in CLASS_NAMES:
        per_class[name] = {
            "precision": None,
            "recall": None,
            "f1": None,
            "support": supports[name],
        }
    return {
        "samples": int(n_total),
        "evaluated": 0,
        "coverage": 0.0 if n_total else None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "positive_class": POSITIVE_CLASS,
        "per_class": per_class,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "confusion_matrix": None,
        "confusion_matrix_labels": list(CLASS_NAMES),
        "avg_inference_time_ms": None,
        "inference_times_ms": [],
        "note": "model unavailable: metrics were not measured",
    }


def _to_index_labels(labels):
    """Coerce string class names to integer indices (pass-through for ints)."""
    return [
        CLASS_TO_IDX.get(label) if isinstance(label, str) else label
        for label in labels
    ]


def _metrics_block(y_true, y_pred, label_names, times_ms) -> dict[str, Any]:
    labels = list(range(len(label_names)))
    y_true = np.asarray(_to_index_labels(y_true))
    y_pred = np.asarray(_to_index_labels(y_pred))
    n_total = int(len(y_true))
    cm = [[0] * len(labels) for _ in labels]
    valid_pairs = [
        (int(yt), int(yp))
        for yt, yp in zip(y_true, y_pred)
        if yt is not None
        and yp is not None
        and isinstance(yt, (int, np.integer))
        and isinstance(yp, (int, np.integer))
        and 0 <= int(yt) < len(labels)
        and 0 <= int(yp) < len(labels)
    ]
    acc = 0.0
    if valid_pairs:
        yt_arr = np.array([p[0] for p in valid_pairs])
        yp_arr = np.array([p[1] for p in valid_pairs])
        cm = confusion_matrix(yt_arr, yp_arr, labels=labels).tolist()
        acc = float(accuracy_score(yt_arr, yp_arr))
    supports = [int((y_true == idx).sum()) for idx in labels]
    per_class = {}
    for idx, name in enumerate(label_names):
        tp = cm[idx][idx]
        fp = int(sum(cm[j][idx] for j in labels if j != idx))
        fn = int(sum(cm[idx][j] for j in labels if j != idx))
        support = int(supports[idx])
        precision = (
            float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        ) if valid_pairs else None
        recall = (float(tp / support) if support > 0 else 0.0) if valid_pairs else None
        f1 = (
            (2 * precision * recall / (precision + recall))
            if precision is not None and (precision + recall) > 0
            else 0.0 if valid_pairs else None
        )
        per_class[name] = {
            "precision": round(precision, 6) if precision is not None else None,
            "recall": round(recall, 6) if recall is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
            "support": support,
        }
    pos_idx = CLASS_TO_IDX[POSITIVE_CLASS]
    tp = cm[pos_idx][pos_idx]
    fp = int(sum(cm[j][pos_idx] for j in labels if j != pos_idx))
    fn = int(sum(cm[pos_idx][j] for j in labels if j != pos_idx))
    binary_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    binary_recall = float(tp / supports[pos_idx]) if supports[pos_idx] > 0 else 0.0
    binary_f1 = (2 * binary_precision * binary_recall / (binary_precision + binary_recall)) \
        if (binary_precision + binary_recall) > 0 else 0.0
    p_vals = [per_class[name]["precision"] for name in label_names]
    r_vals = [per_class[name]["recall"] for name in label_names]
    f_vals = [per_class[name]["f1"] for name in label_names]
    return {
        "samples": n_total,
        "evaluated": len(valid_pairs),
        "coverage": round(len(valid_pairs) / n_total, 6) if n_total else None,
        "accuracy": round(acc, 6) if valid_pairs else None,
        "precision": round(binary_precision, 6) if valid_pairs else None,
        "recall": round(binary_recall, 6) if valid_pairs else None,
        "f1": round(binary_f1, 6) if valid_pairs else None,
        "positive_class": POSITIVE_CLASS,
        "per_class": per_class,
        "macro_precision": round(float(np.mean(p_vals)), 6) if valid_pairs else None,
        "macro_recall": round(float(np.mean(r_vals)), 6) if valid_pairs else None,
        "macro_f1": round(float(np.mean(f_vals)), 6) if valid_pairs else None,
        "confusion_matrix": cm if valid_pairs else None,
        "confusion_matrix_labels": list(label_names),
        "avg_inference_time_ms": round(float(np.mean(times_ms)), 4) if times_ms else None,
        "inference_times_ms": [round(float(t), 4) for t in times_ms],
    }


def _load_agent(agent_cls, artifact_path: Path):
    """Instantiate and warm up an agent; returns (agent, available, note)."""
    agent = agent_cls(artifact_path=artifact_path)
    try:
        agent.warm_up()
    except Exception as exc:  # pragma: no cover - defensive
        return agent, False, f"Failed to load: {exc}"
    if agent.available:
        return agent, True, None
    note = getattr(agent, "load_error", None) or f"Artifact not found: {artifact_path}"
    return agent, False, note


def _lr_predict(image_path: Path) -> dict[str, Any]:
    """Run the legacy LogisticRegression pipeline on one image file."""
    from predict import predict_image_file  # noqa: WPS433

    return predict_image_file(image_path, model_path=LR_ARTIFACT)


def _agent_predictions(agent, image_paths: list[Path]):
    """Run one agent over a list of images; returns (preds, probs, times, errors)."""
    preds, probs, times, errors = [], [], [], []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                out = agent.predict(image)
            if out.get("status") != "ok":
                preds.append(None)
                probs.append(None)
                errors.append(out.get("error") or "unknown error")
                times.append(0.0)
            else:
                preds.append(out["predicted_class"])
                probs.append(out["probabilities"] or {})
                times.append(float(out.get("inference_time_ms") or 0.0))
                errors.append(None)
        except Exception as exc:  # agent-level failure is recorded, never fatal
            preds.append(None)
            probs.append(None)
            errors.append(str(exc))
            times.append(0.0)
    return preds, probs, times, errors



def run_evaluation() -> dict[str, Any]:
    """Evaluate all models and the coordinator on the held-out test split.

    Returns the report dict and writes ``artifacts/evaluation.json``.
    """
    paths, labels, class_names, invalid = list_source_images(DATASET_DIR)
    if class_names != CLASS_NAMES:
        raise ValueError(
            f"Dataset class folders must be ordered as {CLASS_NAMES}; found {class_names}."
        )

    (
        train_paths, val_paths, test_paths,
        train_labels, val_labels, test_labels,
    ) = stratified_split(
        paths, labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    resnet_agent, resnet_ok, resnet_note = _load_agent(
        __import__("models.resnet18_agent", fromlist=["ResNet18Agent"]).ResNet18Agent,
        RESNET18_ARTIFACT,
    )
    mobilenet_agent, mobilenet_ok, mobilenet_note = _load_agent(
        __import__("models.mobilenet_agent", fromlist=["MobileNetV3Agent"]).MobileNetV3Agent,
        MOBILENET_ARTIFACT,
    )
    efficientnet_agent, efficientnet_ok, efficientnet_note = _load_agent(
        __import__("models.efficientnet_agent", fromlist=["EfficientNetB0Agent"]).EfficientNetB0Agent,
        EFFICIENTNET_ARTIFACT,
    )

    from services.decision_coordinator import combine  # noqa: WPS433

    y_true = [class_names[int(lbl)] for lbl in test_labels]
    report_models: dict[str, Any] = {}

    # --- Model A: ResNet-18 ---
    resnet_preds, resnet_probs, resnet_times, resnet_errors = [], [], [], []
    if resnet_ok:
        resnet_preds, resnet_probs, resnet_times, resnet_errors = _agent_predictions(
            resnet_agent, test_paths
        )
    report_models["resnet18"] = {
        "architecture": "ResNet-18 (ImageNet) backbone with a trained two-class classification head (fc layer); backbone frozen",
        "artifact": str(RESNET18_ARTIFACT.relative_to(ROOT)),
        "available": resnet_ok,
        "note": resnet_note,
        **_metrics_block(
            y_true,
            resnet_preds if resnet_ok else [None] * len(y_true),
            class_names,
            resnet_times,
        ),
        "failed_predictions": (
            sum(pred is None for pred in resnet_preds) if resnet_ok else None
        ),
        "inference_errors": [error for error in resnet_errors if error],
    }

    # --- Model B: MobileNetV3 ---
    mobilenet_preds, mobilenet_probs, mobilenet_times, mobilenet_errors = [], [], [], []
    if mobilenet_ok:
        mobilenet_preds, mobilenet_probs, mobilenet_times, mobilenet_errors = _agent_predictions(
            mobilenet_agent, test_paths
        )
    report_models["mobilenetv3"] = {
        "architecture": "MobileNetV3-small (ImageNet) backbone with a trained two-class classification head; backbone frozen",
        "artifact": str(MOBILENET_ARTIFACT.relative_to(ROOT)),
        "available": mobilenet_ok,
        "note": mobilenet_note,
        **_metrics_block(
            y_true,
            mobilenet_preds if mobilenet_ok else [None] * len(y_true),
            class_names,
            mobilenet_times,
        ),
        "failed_predictions": (
            sum(pred is None for pred in mobilenet_preds) if mobilenet_ok else None
        ),
        "inference_errors": [error for error in mobilenet_errors if error],
    }

    # --- Model C: EfficientNet-B0 ---
    efficientnet_preds, efficientnet_probs, efficientnet_times, efficientnet_errors = [], [], [], []
    if efficientnet_ok:
        efficientnet_preds, efficientnet_probs, efficientnet_times, efficientnet_errors = (
            _agent_predictions(efficientnet_agent, test_paths)
        )
    efficientnet_metrics = (
        _metrics_block(
            y_true,
            efficientnet_preds,
            class_names,
            efficientnet_times,
        )
        if efficientnet_ok
        else _unavailable_metrics(len(y_true), y_true)
    )
    report_models["efficientnetb0"] = {
        "architecture": "EfficientNet-B0 (ImageNet) backbone with a trained two-class classification head; backbone frozen",
        "artifact": str(EFFICIENTNET_ARTIFACT.relative_to(ROOT)),
        "available": efficientnet_ok,
        "note": efficientnet_note,
        **efficientnet_metrics,
        "failed_predictions": (
            sum(pred is None for pred in efficientnet_preds) if efficientnet_ok else None
        ),
    }

    # --- Legacy LogisticRegression baseline (optional third row only) ---
    lr_preds, lr_times = [], []
    lr_ok = LR_ARTIFACT.exists()
    if lr_ok:
        for path in test_paths:
            t0 = time.perf_counter()
            try:
                out = _lr_predict(path)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                lr_preds.append(out.get("label"))
                lr_times.append(round(elapsed_ms, 4))
            except Exception as exc:  # pragma: no cover - defensive
                lr_preds.append(None)
                lr_times.append(0.0)
    report_models["logistic_regression_baseline"] = {
        "architecture": "ResNet-18 (ImageNet) backbone -> 512-d feature vector -> pickled scikit-learn LogisticRegression head (legacy baseline)",
        "artifact": str(LR_ARTIFACT.relative_to(ROOT)),
        "available": lr_ok,
        "note": None if lr_ok else f"Artifact not found: {LR_ARTIFACT}",
        **_metrics_block(y_true, lr_preds if lr_ok else [None] * len(y_true), class_names, lr_times),
        "baseline_only": True,
    }

    # --- Decision coordinator on the same test examples ---
    combined_preds, combined_times, combined_errors = [], [], []
    for idx, path in enumerate(test_paths):
        resnet_result = {
            "model_name": "ResNet-18",
            "status": "ok" if resnet_ok and resnet_preds[idx] is not None else "error",
            "predicted_class": resnet_preds[idx] if resnet_ok else None,
            "class_scores": resnet_probs[idx] if resnet_ok else None,
            "inference_time_ms": resnet_times[idx] if resnet_ok else 0.0,
            "error": resnet_errors[idx] if resnet_ok else None,
        }
        mobilenet_result = {
            "model_name": "MobileNetV3",
            "status": "ok" if mobilenet_ok and mobilenet_preds[idx] is not None else "error",
            "predicted_class": mobilenet_preds[idx] if mobilenet_ok else None,
            "class_scores": mobilenet_probs[idx] if mobilenet_ok else None,
            "inference_time_ms": mobilenet_times[idx] if mobilenet_ok else 0.0,
            "error": mobilenet_errors[idx] if mobilenet_ok else None,
        }
        efficientnet_result = {
            "model_name": "EfficientNet-B0",
            "status": "ok" if efficientnet_ok and efficientnet_preds[idx] is not None else "unavailable",
            "predicted_class": efficientnet_preds[idx] if efficientnet_ok else None,
            "class_scores": efficientnet_probs[idx] if efficientnet_ok else None,
            "inference_time_ms": efficientnet_times[idx] if efficientnet_ok else None,
            "error": efficientnet_errors[idx] if efficientnet_ok else efficientnet_note,
        }
        c_t0 = time.perf_counter()
        combined = combine(
            resnet_result,
            mobilenet_result,
            efficientnet_result,
            processing_time_ms=0.0,
        )
        combined_ms = (time.perf_counter() - c_t0) * 1000.0
        combined_preds.append(combined.get("final_prediction"))
        combined_errors.append(combined.get("error"))
        combined_times.append(round(combined_ms, 4))

    report_models["combined_coordinator"] = {
        "architecture": "Decision coordinator requiring unanimity among available ResNet-18, MobileNetV3, and EfficientNet-B0 agents",
        "artifact": None,
        "available": resnet_ok or mobilenet_ok or efficientnet_ok,
        "note": None,
        **_metrics_block(
            y_true,
            combined_preds,
            class_names,
            combined_times,
        ),
        "coordinator_errors": [e for e in combined_errors if e],
        "uncertain_predictions": sum(pred == "uncertain" for pred in combined_preds),
        "failed_predictions": sum(pred is None for pred in combined_preds),
    }

    labels_sorted = sorted(set(int(l) for l in test_labels))
    label_names = [class_names[i] for i in labels_sorted]

    report = {
        "generated_at": now_iso(),
        "seed": RANDOM_SEED,
        "positive_class": POSITIVE_CLASS,
        "dataset": {
            "source_dir": str(DATASET_DIR.relative_to(ROOT)),
            "total_source_images": len(paths),
            "valid_source_images": len(paths),
            "invalid_source_images": len(invalid),
            "invalid_files": invalid,
            "class_names": CLASS_NAMES,
            "class_distribution_all": {
                CLASS_NAMES[i]: int((np.asarray(labels) == i).sum()) for i in range(len(CLASS_NAMES))
            },
            "limitation": "The dataset is very small and imbalanced (5 bird / 55 drone source images). "
                          "Metrics are indicative of the pipeline, not a statistically sound claim of "
                          "real-world performance.",
        },
        "split": {
            "train_samples": len(train_paths),
            "val_samples": len(val_paths),
            "test_samples": len(test_paths),
            "train_distribution": class_counts(train_labels, CLASS_NAMES),
            "val_distribution": class_counts(val_labels, CLASS_NAMES),
            "test_distribution": class_counts(test_labels, CLASS_NAMES),
            "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
            "strategy": "stratified by class and grouped by same-class difference hash (distance <= 3)",
        },
        "models": report_models,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_ARTIFACT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run_evaluation()
    print(json.dumps({
        "generated_at": report["generated_at"],
        "test_samples": report["split"]["test_samples"],
        "resnet18_accuracy": report["models"]["resnet18"].get("accuracy"),
        "mobilenetv3_accuracy": report["models"]["mobilenetv3"].get("accuracy"),
        "combined_accuracy": report["models"]["combined_coordinator"].get("accuracy"),
    }, indent=2))
    print(f"Saved evaluation report: {EVALUATION_ARTIFACT}")
