"""AeroNet AI -- Agent 1: ResNet-18 + Logistic Regression classifier.

Wraps the proven inference pipeline in ``predict.py`` (frozen ResNet-18
ImageNet backbone -> 512-dimensional feature vector -> scikit-learn
``LogisticRegression`` head) and exposes the shared agent interface used by
the decision coordinator.

The backbone and the pickled classification head are cached by ``predict.py``
and loaded once; ``warm_up()`` is invoked at application startup so no request
pays a weight-loading penalty.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image

from predict import DEFAULT_MODEL_PATH, predict_image

CLASS_NAMES = ["bird", "drone"]


class ResNetAgent:
    """ResNet-18 feature extraction + LogisticRegression classification agent.

    Independent of the MobileNetV3 agent: different architecture, different
    training signal, different decision procedure.
    """

    model_name = "ResNet-18 + LogisticRegression"

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self._warmed = False

    def warm_up(self) -> None:
        """Load the artifact and backbone once (application startup)."""
        if self._warmed:
            return
        from predict import get_cached_bundle, get_cached_extractor, get_device

        if self.model_path.exists():
            get_cached_bundle(self.model_path)
            get_cached_extractor(get_device())
        self._warmed = True

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Classify a single RGB image.

        Returns a dict with the shared agent interface:
        ``model_name``, ``predicted_class``, ``class_scores``,
        ``probabilities``, ``inference_time_ms``, ``status`` and ``error``.
        """
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
            if not self.model_path.exists():
                result["error"] = f"ResNet artifact not found: {self.model_path}"
                return result
            t0 = time.perf_counter()
            out = predict_image(image, model_path=self.model_path)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            probs = {str(name): float(prob) for name, prob in (out.get("probabilities") or {}).items()}
            result.update({
                "predicted_class": out.get("label"),
                "class_scores": dict(probs),
                "probabilities": dict(probs),
                "inference_time_ms": round(elapsed_ms, 3),
                "status": "ok",
                "drone_similarity": out.get("drone_similarity"),
            })
        except Exception as exc:  # keep one agent's failure from breaking the other
            result["error"] = f"ResNet inference failed: {exc}"
        return result
