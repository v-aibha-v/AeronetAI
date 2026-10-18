"""AeroNet AI -- training pipeline for the classification heads.

Trains Model A (ResNet-18 classification head) and Model B (MobileNetV3
classification head), and Model C (EfficientNet-B0 classification head) when
ImageNet weights are available, on ``dataset/raw``. Writes the held-out
evaluation report to ``artifacts/evaluation.json``.

Both backbones start from ImageNet pretrained weights and stay frozen; only the
new two-class classification head is trained.  This is the right trade-off for
the small, imbalanced bundled dataset (5 bird / 55 drone images): a full
fine-tune would overfit immediately, while head training is fast on CPU and
reproducible.

Usage:

    python scripts/train_models.py

Artifacts written:

* ``artifacts/resnet18_bird_drone.pt`` -- Model A
* ``artifacts/mobilenet_v3_bird_drone.pt`` -- Model B
* ``artifacts/efficientnet_b0_bird_drone.pt`` -- Model C, when pretrained
  ImageNet weights can be obtained
* ``artifacts/evaluation.json`` -- held-out metrics for all available models, the
  LogisticRegression baseline (third row only), and the coordinator.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.evaluation_service import (  # noqa: E402
    CLASS_NAMES,
    DATASET_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RANDOM_SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    class_counts,
    list_source_images,
    run_evaluation,
    stratified_split,
)
from models.mobilenet_agent import (  # noqa: E402
    CLASS_NAMES as MN_CLASS_NAMES,
    IMAGE_SIZE as MN_IMAGE_SIZE,
    build_mobilenet_v3,
    default_transform as mn_default_transform,
)
from models.efficientnet_agent import (  # noqa: E402
    CLASS_NAMES as EN_CLASS_NAMES,
    IMAGE_SIZE as EN_IMAGE_SIZE,
    build_efficientnet_b0,
    default_transform as en_default_transform,
)
from models.resnet18_agent import (  # noqa: E402
    CLASS_NAMES as RN_CLASS_NAMES,
    IMAGE_SIZE as RN_IMAGE_SIZE,
    build_resnet18,
    default_transform as rn_default_transform,
)

if RN_CLASS_NAMES != CLASS_NAMES or MN_CLASS_NAMES != CLASS_NAMES or EN_CLASS_NAMES != CLASS_NAMES:
    raise RuntimeError("Class names disagree between agents and evaluation service")

ARTIFACTS_DIR = ROOT / "artifacts"
RESNET18_ARTIFACT = ARTIFACTS_DIR / "resnet18_bird_drone.pt"
MOBILENET_ARTIFACT = ARTIFACTS_DIR / "mobilenet_v3_bird_drone.pt"
EFFICIENTNET_ARTIFACT = ARTIFACTS_DIR / "efficientnet_b0_bird_drone.pt"

class TrainDataset(Dataset):
    def __init__(self, image_paths, labels, image_size, augment: bool = False, transform=None) -> None:
        self.image_paths = image_paths
        self.labels = labels
        if transform is not None:
            self.transform = transform
            return
        ops = []
        if augment:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops += [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image

        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), int(self.labels[idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AeroNet classification heads.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per model.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Head learning rate.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience on val loss.")
    parser.add_argument(
        "--efficientnet-only",
        action="store_true",
        help="Train EfficientNet-B0 without replacing the existing ResNet-18 or MobileNetV3 artifacts.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights_for_labels(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long), minlength=num_classes)
    if counts.numel() != num_classes or torch.any(counts == 0):
        raise ValueError("Training split must contain every configured class.")
    return counts.sum() / (num_classes * counts.float())


def _train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
            parameter.requires_grad for parameter in module.parameters(recurse=False)
        ):
            module.eval()
    total = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * images.size(0)
    return total / len(loader.dataset)


def _validate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, targets).item()) * images.size(0)
            correct += int((logits.argmax(1) == targets).sum().item())
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def train_head(
    model: nn.Module,
    train_paths: list[Path],
    train_labels: np.ndarray,
    val_paths: list[Path],
    val_labels: np.ndarray,
    image_size: int,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    device: torch.device,
    train_dataset: Dataset | None = None,
    val_dataset: Dataset | None = None,
) -> dict[str, Any]:
    """Train only the classification head; backbone stays frozen."""
    train_ds = (
        train_dataset
        if train_dataset is not None
        else TrainDataset(train_paths, train_labels, image_size, augment=True)
    )
    val_ds = (
        val_dataset
        if val_dataset is not None
        else TrainDataset(val_paths, val_labels, image_size, augment=False)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = class_weights_for_labels(train_labels, len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=lr)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _validate(model, val_loader, criterion, device)
        history.append({
            "epoch": float(epoch),
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
            "val_acc": round(float(val_acc), 6),
        })
        print(f"  epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_val_loss": round(float(best_val_loss), 6),
        "class_counts": class_counts(train_labels, CLASS_NAMES),
        "class_weights": {
            name: round(float(weight), 6)
            for name, weight in zip(CLASS_NAMES, class_weights.cpu())
        },
    }


def save_artifact(model: nn.Module, path: Path, image_size: int, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "class_names": list(CLASS_NAMES),
        "image_size": int(image_size),
    }
    payload.update(extra)
    torch.save(payload, path)



def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paths, labels, class_names, invalid = list_source_images(DATASET_DIR)
    if class_names != CLASS_NAMES:
        raise ValueError(
            f"Dataset class folders must be ordered as {CLASS_NAMES}; found {class_names}."
        )
    print(f"Source images: {len(paths)} ({class_counts(np.array(labels), CLASS_NAMES)}); invalid: {len(invalid)}")
    for path, err in invalid:
        print(f"  invalid: {path} -- {err}")

    (
        train_paths, val_paths, test_paths,
        train_labels, val_labels, test_labels,
    ) = stratified_split(paths, labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, args.seed)
    print(f"Split: train={len(train_paths)} {class_counts(train_labels, CLASS_NAMES)} | "
          f"val={len(val_paths)} {class_counts(val_labels, CLASS_NAMES)} | "
          f"test={len(test_paths)} {class_counts(test_labels, CLASS_NAMES)}")

    if not args.efficientnet_only:
        # ---------------- Model A: ResNet-18 head ----------------
        print("Training Model A: ResNet-18 classification head...")
        resnet_model = build_resnet18(num_classes=len(CLASS_NAMES)).to(device)
        for param in resnet_model.parameters():
            param.requires_grad = False
        for param in resnet_model.fc.parameters():
            param.requires_grad = True
        resnet_log = train_head(
            resnet_model, train_paths, train_labels, val_paths, val_labels,
            RN_IMAGE_SIZE, args.epochs, args.lr, args.batch_size, args.patience, device,
        )
        save_artifact(
            resnet_model, RESNET18_ARTIFACT, RN_IMAGE_SIZE,
            {"arch": "resnet18", "backbone": "resnet18_imagenet", "trainable": "fc_head_only",
             "training": resnet_log},
        )
        print(f"Saved {RESNET18_ARTIFACT}")

        # ---------------- Model B: MobileNetV3 head ----------------
        print("Training Model B: MobileNetV3-small classification head...")
        mobilenet_model = build_mobilenet_v3(num_classes=len(CLASS_NAMES)).to(device)
        for param in mobilenet_model.features.parameters():
            param.requires_grad = False
        for param in mobilenet_model.classifier.parameters():
            param.requires_grad = True
        mobilenet_log = train_head(
            mobilenet_model, train_paths, train_labels, val_paths, val_labels,
            MN_IMAGE_SIZE, args.epochs, args.lr, args.batch_size, args.patience, device,
        )
        save_artifact(
            mobilenet_model, MOBILENET_ARTIFACT, MN_IMAGE_SIZE,
            {"arch": "mobilenet_v3_small", "backbone": "mobilenet_v3_small_imagenet",
             "trainable": "classifier_head_only", "training": mobilenet_log},
        )
        print(f"Saved {MOBILENET_ARTIFACT}")

    # ---------------- Model C: EfficientNet-B0 head ----------------
    print("Training Model C: EfficientNet-B0 classification head...")
    try:
        efficientnet_model = build_efficientnet_b0(
            num_classes=len(CLASS_NAMES),
            pretrained=True,
        ).to(device)
    except Exception as exc:
        efficientnet_model = None
        print(f"EfficientNet-B0 unavailable; pretrained ImageNet weights could not be loaded: {exc}")

    if efficientnet_model is not None:
        for param in efficientnet_model.features.parameters():
            param.requires_grad = False
        for param in efficientnet_model.classifier.parameters():
            param.requires_grad = True
        efficientnet_train_ds = TrainDataset(
            train_paths,
            train_labels,
            EN_IMAGE_SIZE,
            transform=en_default_transform(augment=True),
        )
        efficientnet_val_ds = TrainDataset(
            val_paths,
            val_labels,
            EN_IMAGE_SIZE,
            transform=en_default_transform(),
        )
        # Reuse the shared training loop with the model's expected transforms.
        efficientnet_log = train_head(
            efficientnet_model,
            train_paths,
            train_labels,
            val_paths,
            val_labels,
            EN_IMAGE_SIZE,
            args.epochs,
            args.lr,
            args.batch_size,
            args.patience,
            device,
            train_dataset=efficientnet_train_ds,
            val_dataset=efficientnet_val_ds,
        )
        save_artifact(
            efficientnet_model,
            EFFICIENTNET_ARTIFACT,
            EN_IMAGE_SIZE,
            {
                "arch": "efficientnet_b0",
                "backbone": "efficientnet_b0_imagenet",
                "trainable": "classifier_head_only",
                "training": efficientnet_log,
            },
        )
        print(f"Saved {EFFICIENTNET_ARTIFACT}")

    # ---------------- Evaluation on the held-out test split ----------------
    print("Evaluating on the held-out test split...")
    report = run_evaluation()
    print(json.dumps({
        "test_samples": report["split"]["test_samples"],
        "test_distribution": report["split"]["test_distribution"],
        "resnet18_accuracy": report["models"]["resnet18"].get("accuracy"),
        "mobilenetv3_accuracy": report["models"]["mobilenetv3"].get("accuracy"),
        "efficientnetb0_accuracy": report["models"]["efficientnetb0"].get("accuracy"),
        "combined_accuracy": report["models"]["combined_coordinator"].get("accuracy"),
    }, indent=2))
    print(f"Saved evaluation report: {ARTIFACTS_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()
