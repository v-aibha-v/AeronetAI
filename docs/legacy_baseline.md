Legacy baseline reference

`models/resnet_agent.py` with `predict.py` implements the older
LogisticRegression baseline: ResNet-18 backbone features fed into a
pickled scikit-learn head (`artifacts/drone_bird_feature_lr.pkl`).

It is kept for comparison only. It appears as a third row in the
evaluation table and is not wired into the coordinator or `/api/detect`.

The `drone_centroid.npy` artifact supports an extra drone-similarity
score in `predict_image`, which is informational and separate from the
coordinator confidence.
