Classification agents overview

Three independent image agents feed the coordinator. They share the
`bird` / `drone` class list and the `predict(image)` result shape, but
differ in architecture and input size.

- Model A, ResNet-18 (`models/resnet18_agent.py`): torchvision ResNet-18
  backbone, 160 px input, artifact
  `artifacts/resnet18_bird_drone.pt`.
- Model B, MobileNetV3-small (`models/mobilenet_agent.py`): torchvision
  MobileNetV3-small backbone, 224 px input, artifact
  `artifacts/mobilenet_v3_bird_drone.pt`.
- Model C, EfficientNet-B0 (`models/efficientnet_agent.py`): torchvision
  EfficientNet-B0 backbone, 224 px input with ImageNet weight transforms,
  artifact `artifacts/efficientnet_b0_bird_drone.pt`.

A legacy LogisticRegression baseline (`models/resnet_agent.py` with
`predict.py`, artifact `artifacts/drone_bird_feature_lr.pkl`) appears as
an evaluation-table row only. It is not part of the coordinator.

Each agent loads its artifact once at startup (`warm_up`) and reports
`available`, `status`, and `load_error` through `GET /api/models`.
