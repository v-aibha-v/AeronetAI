"""EfficientNet-B0 bird/drone classifier backed by a trained project artifact."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

CLASS_NAMES = ["bird", "drone"]
IMAGE_SIZE = 224
DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent / "artifacts" / "efficientnet_b0_bird_drone.pt"
)


def build_efficientnet_b0(
    num_classes: int = len(CLASS_NAMES),
    *,
    pretrained: bool = False,
) -> torch.nn.Module:
    """Build EfficientNet-B0; pretrained weights are requested only for training."""
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model


def default_transform(*, augment: bool = False):
    """Use the resize, crop, interpolation, and normalization for ImageNet weights."""
    weights_transform = EfficientNet_B0_Weights.DEFAULT.transforms()
    operations = []
    if augment:
        operations.append(transforms.RandomHorizontalFlip(p=0.5))
    operations.extend([
        transforms.Resize(
            weights_transform.resize_size,
            interpolation=weights_transform.interpolation,
            antialias=weights_transform.antialias,
        ),
        transforms.CenterCrop(weights_transform.crop_size),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(
            mean=weights_transform.mean,
            std=weights_transform.std,
        ),
    ])
    return transforms.Compose(operations)


class EfficientNetB0Agent:
    """EfficientNet-B0 agent; unavailable until a compatible trained artifact exists."""

    model_name = "EfficientNet-B0"

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        self.artifact_path = (
            Path(artifact_path) if artifact_path is not None else DEFAULT_ARTIFACT_PATH
        )
        self.model: torch.nn.Module | None = None
        self.device = torch.device("cpu")
        self.class_names = list(CLASS_NAMES)
        self._warmed = False
        self.load_error: str | None = None

    def warm_up(self) -> None:
        """Load a saved two-class model once without requesting network weights."""
        if self._warmed:
            return
        self._warmed = True
        if not self.artifact_path.exists():
            self.load_error = f"Trained EfficientNet-B0 artifact not found: {self.artifact_path}"
            return
        try:
            checkpoint = torch.load(
                self.artifact_path,
                map_location=self.device,
                weights_only=True,
            )
            if checkpoint.get("arch") != "efficientnet_b0":
                raise ValueError("checkpoint architecture is not efficientnet_b0")
            class_names = checkpoint.get("class_names")
            if class_names != CLASS_NAMES:
                raise ValueError(f"checkpoint classes must be {CLASS_NAMES}")
            model = build_efficientnet_b0(num_classes=len(class_names), pretrained=False)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(self.device).eval()
            self.class_names = list(class_names)
            self.model = model
        except Exception as exc:
            self.load_error = f"Failed to load EfficientNet-B0 artifact: {exc}"
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, image: Image.Image) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model_name": self.model_name,
            "predicted_class": None,
            "class_scores": None,
            "probabilities": None,
            "inference_time_ms": None,
            "status": "unavailable" if self.model is None else "error",
            "error": self.load_error,
        }
        try:
            if self.model is None:
                if self.load_error is None:
                    result["error"] = "EfficientNet-B0 is not loaded."
                return result
            inputs = default_transform()(image.convert("RGB")).unsqueeze(0).to(self.device)
            started = time.perf_counter()
            with torch.inference_mode():
                probabilities = torch.softmax(self.model(inputs), dim=1).cpu().numpy()[0]
            scores = {
                name: float(probability)
                for name, probability in zip(self.class_names, probabilities)
            }
            result.update({
                "predicted_class": self.class_names[int(probabilities.argmax())],
                "class_scores": scores,
                "probabilities": scores,
                "inference_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "status": "ok",
                "error": None,
            })
        except Exception as exc:
            result["error"] = f"EfficientNet-B0 inference failed: {exc}"
        return result
