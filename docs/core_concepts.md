Core concepts
This project classifies an uploaded still image as a bird or a drone.
It runs three independent deep learning agents. A coordinator fuses their outputs into a single decision.

Agents:
ResNet-18 plus LogisticRegression: a baseline shown in the evaluation table.
MobileNetV3-Small: an active agent.
EfficientNet-B0: an active agent. It requires pretrained ImageNet weights.

Coordinator:
The coordinator compares the available agents. All available agents must agree before the shared class is returned.
If the available agents disagree, the result is uncertain.
If only one agent works, that agent is used alone.
If no agent works, the result is error.

Why disagreement is honest:
Forcing a guess when agents disagree would hide model uncertainty.
The coordinator returns uncertain instead. This guarantees that a weak result is labelled uncertain rather than reported with false confidence.

Result fields:
final_prediction.
agreement_status.
agent_results.
decision_reason.
confidence.
processing_time_ms.

