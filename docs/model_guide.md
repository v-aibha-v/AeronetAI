Model guide
The models folder contains the agent implementations. Checkpoints live in artifacts. The agent files contain no weights.

Agent interface:
Every agent exposes warm_up, available, and predict.
warm_up: load the model and backbone once, at startup.
available: whether a trained checkpoint exists.
predict: classify a single RGB image and return model_name, predicted_class, class_scores, probabilities, inference_time_ms, status, and error.

Models:
ResNet-18 plus LogisticRegression: a baseline agent. It is not part of the coordinator. It is shown in the evaluation table only.
MobileNetV3-Small: an active agent.
EfficientNet-B0: an active agent. It requires pretrained ImageNet weights.

Training:
All active agents use a frozen ImageNet backbone plus a trained two-class head. Head-only training keeps the small imbalanced dataset from overfitting and runs fast on CPU.

Honest availability:
If a checkpoint is missing, the agent reports unavailable. The coordinator excludes unavailable agents and returns error only when no agent is usable.

