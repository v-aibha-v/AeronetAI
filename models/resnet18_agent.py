"""AeroNet AI -- Model A: ResNet-18 standalone image classifier.

A torchvision ``resnet18`` backbone pretrained on ImageNet with its
classification head replaced for the two target classes ``bird`` and ``drone``.
Only the classification head is trained; the backbone stays frozen, which works
well on small datasets and keeps CPU training fast.

The trained artifact (state dict + class mapping + image size) is saved to
``artifacts/resnet18_bird_drone.pt`` by ``scripts/train_models.py`` and loaded
once at application startup.  CPU execution is fully supported.

This agent is independent of:

* the LogisticRegression baseline in ``models/resnet_agent.py`` / ``predict.py``
  (ResNet-18 backbone -> feature vector -> pickled scikit-learn head), and
* the MobileNetV3 agent in ``models/mobilenet_agent.py`` (different architecture,
  different training signal, different decision procedure).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

CLASS_NAMES = ["bird", "drone"]
IMAGE_SIZE = 160
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "resnet18_bird_drone.pt"


def build_resnet18(num_classes: int = 2, *, pretrained: bool = True) -> torch.nn.Module:
    """Build a ResNet-18 with its classification head adapted to ``num_classes``."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    backbone = models.resnet18(weights=weights)
    in_features = backbone.fc.in_features
    backbone.fc = torch.nn.Linear(in_features, num_classes)
    return backbone


def default_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ResNet18Agent:
    """ResNet-18 bird/vs drone classification agent."""

    model_name = "ResNet-18"

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        self.artifact_path = Path(artifact_path) if artifact_path is not None else DEFAULT_ARTIFACT_PATH
        self.model: torch.nn.Module | None = None
        self.device = torch.device("cpu")
        self.class_names = list(CLASS_NAMES)
        self.image_size = IMAGE_SIZE
        self._warmed = False
        self.load_error: str | None = None

    def warm_up(self) -> None:
        """Load the trained artifact once (application startup)."""
        if self._warmed:
            return
        self._warmed = True
        if not self.artifact_path.exists():
            self.model = None
            self.load_error = f"Trained ResNet-18 artifact not found: {self.artifact_path}"
            return
        try:
            checkpoint = torch.load(self.artifact_path, map_location=self.device, weights_only=True)
            if checkpoint.get("arch") != "resnet18":
                raise ValueError("checkpoint architecture is not resnet18")
            class_names = checkpoint.get("class_names")
            if class_names != CLASS_NAMES:
                raise ValueError(f"checkpoint classes must be {CLASS_NAMES}")
            self.class_names = list(class_names)
            self.image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
            model = build_resnet18(num_classes=len(self.class_names), pretrained=False)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(self.device).eval()
            self.model = model
        except Exception as exc:  # keep the service up; coordinator reports degraded mode
            self.model = None
            self.load_error = f"Failed to load ResNet-18 artifact: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Classify a single RGB image with the shared agent interface."""
        result: dict[str, Any] = {
            "model_name": self.model_name,
            "predicted_class": None,
            "class_scores": None,
            "probabilities": None,
            "inference_time_ms": None,
            "status": "error",
            "error": None,
        }
        try:
            if self.model is None:
                result["error"] = self.load_error or "ResNet-18 model is not loaded."
                return result
            image = image.convert("RGB")
            transform = default_transform(self.image_size)
            inputs = transform(image).unsqueeze(0).to(self.device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(inputs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            scores = {name: float(prob) for name, prob in zip(self.class_names, probs)}
            best_idx = int(probs.argmax())
            result.update({
                "predicted_class": self.class_names[best_idx],
                "class_scores": scores,
                "probabilities": scores,
                "inference_time_ms": round(elapsed_ms, 3),
                "status": "ok",
            })
        except Exception as exc:
            result["error"] = f"ResNet-18 inference failed: {exc}"
        return result
