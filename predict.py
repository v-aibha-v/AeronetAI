from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "drone_bird_feature_lr.pkl"
DRONE_CENTROID_PATH = ROOT / "artifacts" / "drone_centroid.npy"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


_MODEL_CACHE: dict[str, Any] = {}
_EXTRACTOR_CACHE: dict[str, Any] = {}
_DRONE_CENTROID: np.ndarray | None = None


def get_drone_centroid() -> np.ndarray | None:
    global _DRONE_CENTROID
    if _DRONE_CENTROID is None and DRONE_CENTROID_PATH.exists():
        _DRONE_CENTROID = np.load(DRONE_CENTROID_PATH)
    return _DRONE_CENTROID


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    with Path(model_path).open("rb") as fh:
        return pickle.load(fh)


def get_cached_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    key = str(model_path)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_bundle(model_path)
    return _MODEL_CACHE[key]


def build_feature_extractor(device: torch.device) -> torch.nn.Module:
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    backbone.to(device)
    return backbone


def get_cached_extractor(device: torch.device) -> torch.nn.Module:
    key = str(device)
    if key not in _EXTRACTOR_CACHE:
        _EXTRACTOR_CACHE[key] = build_feature_extractor(device)
    return _EXTRACTOR_CACHE[key]


def prepare_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in getattr(image, "info", {})):
        rgba = image.convert("RGBA")
        alpha = rgba.split()[-1]
        bg = Image.new("RGB", rgba.size, (200, 220, 240))
        bg.paste(rgba, mask=alpha)
        return bg
    return image.convert("RGB")


def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transform(prepare_rgb(image)).unsqueeze(0)



def predict_image(
    image: Image.Image,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    bundle = get_cached_bundle(model_path)
    classifier = bundle["classifier"]
    class_names = bundle["class_names"]
    image_size = bundle["image_size"]

    device = get_device()
    extractor = get_cached_extractor(device)
    inputs = preprocess(image, image_size).to(device)

    with torch.inference_mode():
        features = extractor(inputs).cpu().numpy()

    probs = classifier.predict_proba(features)[0]
    best_idx = int(np.argmax(probs))

    drone_sim = 0.0
    centroid = get_drone_centroid()
    if centroid is not None:
        feat_vec = features[0]
        norm = float(np.linalg.norm(feat_vec))
        if norm > 0:
            drone_sim = float(np.dot(feat_vec / norm, centroid))

    return {
        "label": class_names[best_idx],
        "confidence": float(probs[best_idx]),
        "probabilities": {name: float(prob) for name, prob in zip(class_names, probs)},
        "drone_similarity": drone_sim,
    }


def predict_image_file(image_path: str | Path, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    image = Image.open(image_path)
    return predict_image(image, model_path=model_path)
