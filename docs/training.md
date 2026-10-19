Training pipeline

`scripts/train_models.py` trains only the two-class classification heads.
All ImageNet backbones stay frozen.

- Source data: `dataset/raw` with `bird` and `drone` folders.
- Model A (ResNet-18) and Model B (MobileNetV3-small) are always trained;
  Model C (EfficientNet-B0) trains when its ImageNet weights can be
  obtained.
- Training uses horizontal-flip augmentation on the train split and
  plain resize/normalize transforms on validation.
- Outputs: `artifacts/resnet18_bird_drone.pt`,
  `artifacts/mobilenet_v3_bird_drone.pt`,
  `artifacts/efficientnet_b0_bird_drone.pt` when available, plus
  `artifacts/evaluation.json` from the held-out test split.
- Head-only training is the documented trade-off for the small bundled
  dataset (5 bird / 55 drone source images): full fine-tuning would
  overfit immediately, while head training stays fast on CPU and
  reproducible.

Run with `python scripts/train_models.py` or `POST /api/evaluate`.
