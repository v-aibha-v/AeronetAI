Image preprocessing

All agents expect RGB still images normalized with ImageNet statistics.

- Mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- ResNet-18 resizes to 160 x 160; MobileNetV3-small resizes to
  224 x 224; EfficientNet-B0 applies the ImageNet weight transforms
  (resize, center crop, tensor conversion, normalization).
- Inputs are converted to RGB first; RGBA/LA/P images with transparency
  are composited before conversion in the baseline path.
- `ImageFile.LOAD_TRUNCATED_IMAGES = True` lets truncated files decode
  instead of raising, so validation still reports them accurately.
- Preprocessing parameters come from the saved artifacts, never from
  test data, keeping evaluation honest.
