Troubleshooting
Model checkpoint missing: the application reports models as unavailable when a trained checkpoint is not in artifacts. The coordinator returns error only when no agent is usable. To run real inference and evaluation, train the models first with python scripts/train_models.py.

Slow first run: warm_up loads the model and backbone once at startup. The first request may be slower because of checkpoint loading. Later requests reuse the cached model.

CPU only: the project has no GPU requirement. Use the CPU build of PyTorch from requirements.txt. Run with python app.py.

Test failures: some tests load model artifacts and require the checkpoints to exist. Generate them with the training script, or run the fast unit tests with python -m pytest -m not slow -q.

Marker errors: if pytest reports that a marker is not registered, run the tests with the project pytest.ini addopts: python -m pytest -q

