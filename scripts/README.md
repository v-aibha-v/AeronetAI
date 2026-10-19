Training script
This folder contains the training and evaluation pipeline entry point.

## train_models.py
Trains the three agent classification heads and evaluates them on a held-out split.


Usage:
python scripts/train_models.py

Options:
- --epochs: training epochs per model, default 20.
- --lr: head learning rate, default 0.001.
- --batch-size: batch size, default 32.
- --patience: early stopping patience on validation loss, default 3.

Artifacts written:
- artifacts/resnet18_bird_drone.pt: Model A.
- artifacts/mobilenet_v3_bird_drone.pt: Model B.
- artifacts/efficientnet_b0_bird_drone.pt: Model C, when pretrained ImageNet weights can be loaded.
- artifacts/evaluation.json: held-out metrics for every available model, the LogisticRegression baseline, and the coordinator.

