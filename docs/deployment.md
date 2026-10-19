Local deployment

Requirements:

- Python 3.10 or newer.
- Dependencies from `requirements.txt`, including the CPU PyTorch
  wheels referenced by the extra index URL.

Steps:

1. Create and activate a virtual environment.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Train the classification heads: `python scripts/train_models.py`.
4. Start the API: `python app.py`.
5. Open `http://127.0.0.1:8000/` in a browser.

The server runs with `uvicorn app:app` on host `0.0.0.0`, port 8000,
without reload mode.

Render deployment:

`render.yaml` defines a Python web service named `aeronet` with:

- Build command: `pip install -r requirements.txt`.
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`.
- `PYTHON_VERSION` set to `3.11.9`.
