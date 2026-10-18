"""AeroNet AI -- Decision Coordinator (evidence aggregation agent).

Receives the independent results of ResNet-18, MobileNetV3, and EfficientNet-B0,
checks their validity, compares their predictions,
and emits a single explainable decision.

Decision strategy (deterministic, documented; no learned weights):

* If all available agents predict the same class -> that class. No majority
  voting is used; if two or more valid predictions conflict, return
  ``uncertain``.
* Missing or failed agents are excluded from comparison, not treated as votes.
* If no agent works -> status ``error``, no fabricated prediction.

Agreement of two models never guarantees correctness; every result carries an
explainable ``decision_reason``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

AGREE = "agree"
DISAGREE = "disagree"
SINGLE_MODEL = "single_model"
ERROR = "error"
UNCERTAIN = "uncertain"

VALID_CLASSES = ("bird", "drone")

@dataclass
class CoordinatorResult:
    final_prediction: str | None
    agreement_status: str
    agent_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    decision_reason: str = ""
    confidence: float | None = None
    processing_time_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_prediction": self.final_prediction,
            "agreement_status": self.agreement_status,
            "agent_results": self.agent_results,
            "decision_reason": self.decision_reason,
            "confidence": None if self.confidence is None else round(float(self.confidence), 4),
            "processing_time_ms": round(float(self.processing_time_ms), 3),
            "error": self.error,
        }
def _agent_summary(result: dict[str, Any], model_key: str) -> dict[str, Any]:
    return {
        "model": result.get("model_name", model_key),
        "status": result.get("status", "error"),
        "predicted_class": result.get("predicted_class"),
        "class_scores": result.get("class_scores"),
        "inference_time_ms": result.get("inference_time_ms"),
        "error": result.get("error"),
    }


def _class_score(result: dict[str, Any], cls: str) -> float | None:
    scores = result.get("class_scores") or {}
    score = scores.get(cls)
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return float(score)
    return None


def combine(
    resnet_result: dict[str, Any],
    mobilenet_result: dict[str, Any],
    efficientnet_result: dict[str, Any] | None = None,
    *,
    processing_time_ms: float = 0.0,
) -> dict[str, Any]:
    """Return a class only when every usable agent agrees."""
    if efficientnet_result is None:
        efficientnet_result = {
            "model_name": "EfficientNet-B0",
            "status": "unavailable",
            "predicted_class": None,
            "class_scores": None,
            "inference_time_ms": None,
            "error": "EfficientNet-B0 result was not provided.",
        }
    results = {
        "resnet18": resnet_result,
        "mobilenetv3": mobilenet_result,
        "efficientnetb0": efficientnet_result,
    }
    agent_results = {
        key: _agent_summary(result, key) for key, result in results.items()
    }
    valid_results = [
        result for result in results.values()
        if result.get("status") == "ok" and result.get("predicted_class") in VALID_CLASSES
    ]

    if not valid_results:
        return CoordinatorResult(
            final_prediction=None,
            agreement_status=ERROR,
            agent_results=agent_results,
            decision_reason="all_agents_unavailable_or_failed",
            confidence=None,
            processing_time_ms=processing_time_ms,
            error="No classification agent produced a valid prediction.",
        ).to_dict()

    predictions = {result["predicted_class"] for result in valid_results}
    if len(predictions) > 1:
        return CoordinatorResult(
            final_prediction=UNCERTAIN,
            agreement_status=DISAGREE,
            agent_results=agent_results,
            decision_reason="models_disagree",
            confidence=None,
            processing_time_ms=processing_time_ms,
        ).to_dict()

    prediction = valid_results[0]["predicted_class"]
    scores = [
        score for result in valid_results
        if (score := _class_score(result, prediction)) is not None
    ]
    confidence = sum(scores) / len(scores) if scores else None
    agreement_status = AGREE if len(valid_results) > 1 else SINGLE_MODEL
    reason = (
        "available_models_agree"
        if len(valid_results) > 1
        else "single_model_available"
    )
    return CoordinatorResult(
        final_prediction=prediction,
        agreement_status=agreement_status,
        agent_results=agent_results,
        decision_reason=reason,
        confidence=confidence,
        processing_time_ms=processing_time_ms,
    ).to_dict()
