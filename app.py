"""AeroNet AI -- image-only bird vs drone multi-model classification backend.

FastAPI application with three image-classification agents:

* Model A -- ResNet-18 (ImageNet) backbone with a trained two-class head
  (``models/resnet18_agent.py``)
* Model B -- MobileNetV3-small (ImageNet) backbone with a trained two-class head
  (``models/mobilenet_agent.py``)
* Model C -- EfficientNet-B0 with a trained two-class head
  (``models/efficientnet_agent.py``)

A legacy LogisticRegression baseline (``predict.py``) is preserved as an
optional third row in the evaluation comparison table only; it is not part of
the coordinator.

A deterministic decision coordinator (``services/decision_coordinator.py``)
compares the available agents and returns a single ``bird`` / ``drone`` / ``uncertain`` /
error decision.

This application handles still images only. All video, webcam, live-stream and
object-detection functionality has been removed.

Endpoints:
* GET  /                         -- home page (HTML)
* POST /api/detect               -- classify an uploaded image (available agents +
                                    coordinator), JSON
* GET  /api/models               -- per-agent availability/status
* GET  /api/evaluation           -- saved held-out evaluation metrics (JSON)
* POST /api/evaluate             -- (re)run training + evaluation
* GET  /api/manifest             -- dataset manifest (JSON)
* GET  /api/stats                -- dashboard telemetry from prediction history
* GET  /api/history.json         -- prediction history (JSON)
"""

from __future__ import annotations

import base64
import builtins
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from models.efficientnet_agent import EfficientNetB0Agent
from models.mobilenet_agent import MobileNetV3Agent
from models.resnet18_agent import ResNet18Agent
from models.resnet_agent import ResNetAgent  # legacy baseline (evaluation table only)
from services.decision_coordinator import combine
from services.evaluation_service import DATASET_DIR, EVALUATION_ARTIFACT, run_evaluation

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
HISTORY_PATH = ARTIFACTS_DIR / "prediction_history.json"
EVALUATION_ARTIFACT = ARTIFACTS_DIR / "evaluation.json"
builtins.HISTORY_PATH = HISTORY_PATH

app = FastAPI(title="AeroNet AI -- Multi-Model Bird vs Drone Classification")
app.mount("/samples", StaticFiles(directory=str(ROOT / "samples")), name="samples")
templates = Jinja2Templates(directory=str(ROOT / "templates"))

# ---------------------------------------------------------------------------
# Model loading -- once at application startup, reused for every request
# ---------------------------------------------------------------------------

RESNET_AGENT = ResNet18Agent()
MOBILENET_AGENT = MobileNetV3Agent()
EFFICIENTNET_AGENT = EfficientNetB0Agent()
BASELINE_AGENT = ResNetAgent()  # LogisticRegression baseline, not in the coordinator

MODEL_STATE: dict[str, dict[str, Any]] = {}

try:
    RESNET_AGENT.warm_up()
    MODEL_STATE["resnet18"] = {
        "name": RESNET_AGENT.model_name,
        "available": RESNET_AGENT.available,
        "status": "ok" if RESNET_AGENT.available else "unavailable",
        "note": None if RESNET_AGENT.available else f"Artifact not found: {RESNET_AGENT.artifact_path}",
    }
except Exception as exc:  # never let model startup crash the server
    MODEL_STATE["resnet18"] = {
        "name": "ResNet-18",
        "available": False,
        "status": "error",
        "note": f"Failed to load: {exc}",
    }

try:
    MOBILENET_AGENT.warm_up()
    MODEL_STATE["mobilenetv3"] = {
        "name": MOBILENET_AGENT.model_name,
        "available": MOBILENET_AGENT.available,
        "status": "ok" if MOBILENET_AGENT.available else "unavailable",
        "note": None if MOBILENET_AGENT.available else f"Artifact not found: {MOBILENET_AGENT.artifact_path}",
    }
except Exception as exc:
    MODEL_STATE["mobilenetv3"] = {
        "name": "MobileNetV3",
        "available": False,
        "status": "error",
        "note": f"Failed to load: {exc}",
    }

try:
    EFFICIENTNET_AGENT.warm_up()
    MODEL_STATE["efficientnetb0"] = {
        "name": EFFICIENTNET_AGENT.model_name,
        "available": EFFICIENTNET_AGENT.available,
        "status": "ok" if EFFICIENTNET_AGENT.available else "unavailable",
        "note": EFFICIENTNET_AGENT.load_error,
    }
except Exception as exc:
    MODEL_STATE["efficientnetb0"] = {
        "name": EFFICIENTNET_AGENT.model_name,
        "available": False,
        "status": "error",
        "note": f"Failed to load: {exc}",
    }

try:
    BASELINE_AGENT.warm_up()
    MODEL_STATE["logistic_regression_baseline"] = {
        "name": BASELINE_AGENT.model_name,
        "available": BASELINE_AGENT.available,
        "status": "ok" if BASELINE_AGENT.available else "unavailable",
        "note": None if BASELINE_AGENT.available else f"Artifact not found: {BASELINE_AGENT.model_path}",
    }
except Exception as exc:
    MODEL_STATE["logistic_regression_baseline"] = {
        "name": "ResNet-18 + LogisticRegression (baseline)",
        "available": False,
        "status": "error",
        "note": f"Failed to load: {exc}",
    }

# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # 16 MiB


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def secure_filename(filename: str) -> str:
    name = Path(filename or "").name.replace(" ", "_")
    return f"{uuid.uuid4().hex}_{name}"


def temp_path(suffix: str) -> Path:
    import tempfile

    fd, path = tempfile.mkstemp(suffix=suffix, prefix="aeronet_")
    os.close(fd)
    return Path(path)


def safe_cleanup(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def validate_upload(file: UploadFile) -> tuple[bool, str, Path | None]:
    """Validate an image upload: non-empty, allowed type, decodable image."""
    if not file.filename:
        return False, "No file selected.", None
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}.", None
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        return False, f"File too large ({file.size} bytes). Max {MAX_UPLOAD_BYTES} bytes.", None
    temp = temp_path(suffix)
    try:
        content = file.file.read()
        if len(content) == 0:
            safe_cleanup(temp)
            return False, "Uploaded file is empty.", None
        temp.write_bytes(content)
        # Prove it is a real, decodable image (not a renamed text file).
        with Image.open(temp) as im:
            im.load()
    except Exception as exc:
        safe_cleanup(temp)
        return False, f"Could not decode image: {exc}", None
    return True, "", temp


def image_to_data_uri(image: Image.Image) -> str | None:
    """Encode an image as a JPEG data URI for the frontend preview."""
    try:
        rgb = image.convert("RGB") if image.mode != "RGB" else image
    except Exception:
        return None
    buf = BytesIO()
    rgb.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


# ---------------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------------

def load_history() -> list[dict[str, Any]]:
    try:
        if HISTORY_PATH.exists():
            history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            return history if isinstance(history, list) else []
    except Exception:
        pass
    return []


def record_history(entry: dict[str, Any]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        history = load_history()
        history.insert(0, entry)
        HISTORY_PATH.write_text(json.dumps(history[:1000], indent=2), encoding="utf-8")
    except Exception:
        pass


def load_manifest() -> dict[str, Any]:
    manifest_path = ROOT / "dataset" / "manifest.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request, "init_history": json.dumps(load_history())},
    )


@app.post("/api/detect")
async def api_detect(image: UploadFile = File(...)) -> JSONResponse:
    """Classify one uploaded image with all agents and the coordinator."""
    overall_t0 = time.perf_counter()

    ok, error, temp = validate_upload(image)
    if not ok:
        return JSONResponse({"error": error}, status_code=400)

    try:
        with Image.open(temp) as pil_image:
            pil_image.load()

            resnet_out = RESNET_AGENT.predict(pil_image)
            mobilenet_out = MOBILENET_AGENT.predict(pil_image)
            efficientnet_out = EFFICIENTNET_AGENT.predict(pil_image)

            c_t0 = time.perf_counter()
            combined = combine(
                resnet_out,
                mobilenet_out,
                efficientnet_out,
                processing_time_ms=0.0,
            )
            combined_ms = (time.perf_counter() - c_t0) * 1000.0

        overall_ms = (time.perf_counter() - overall_t0) * 1000.0

        uploaded_uri = image_to_data_uri(pil_image)
    except Exception as exc:
        safe_cleanup(temp)
        return JSONResponse({"error": f"Classification failed: {exc}"}, status_code=500)
    finally:
        safe_cleanup(temp)

    decision = combined.get("final_prediction")
    if decision is None:
        decision = "error"
    confidence = combined.get("confidence")
    reason = combined.get("decision_reason", "")

    entry = {
        "timestamp": now_iso(),
        "file_name": secure_filename(image.filename or "upload"),
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
        "models": {
            "resnet18": resnet_out.get("predicted_class"),
            "mobilenetv3": mobilenet_out.get("predicted_class"),
            "efficientnetb0": efficientnet_out.get("predicted_class"),
        },
    }
    record_history(entry)

    return JSONResponse({
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
        "processing_time_ms": round(overall_ms, 3),
        "models": {
            "resnet18": resnet_out,
            "mobilenetv3": mobilenet_out,
            "efficientnetb0": efficientnet_out,
        },
        "result": combined,
        "uploaded_image": uploaded_uri,
    })


@app.get("/api/models")
async def api_models() -> JSONResponse:
    return JSONResponse(MODEL_STATE)


@app.get("/api/evaluation")
async def api_evaluation() -> JSONResponse:
    if EVALUATION_ARTIFACT.exists():
        try:
            return JSONResponse(json.loads(EVALUATION_ARTIFACT.read_text(encoding="utf-8")))
        except Exception:
            pass
    return JSONResponse({
        "available": False,
        "message": "Evaluation metrics have not been computed yet. Run POST /api/evaluate or python scripts/train_models.py.",
    }, status_code=404)


@app.post("/api/evaluate")
async def api_evaluate() -> JSONResponse:
    """(Re)run classification-head training and held-out evaluation."""
    try:
        completed = subprocess.run(
            ["python", str(ROOT / "scripts" / "train_models.py")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        return JSONResponse({
            "error": "Training/evaluation failed",
            "stdout": exc.stdout[-2000:] if exc.stdout else None,
            "stderr": exc.stderr[-2000:] if exc.stderr else None,
        }, status_code=500)
    except Exception as exc:  # pragma: no cover - defensive
        return JSONResponse({"error": f"Training/evaluation failed: {exc}"}, status_code=500)

    try:
        report = json.loads(EVALUATION_ARTIFACT.read_text(encoding="utf-8"))
    except Exception:
        report = {"note": "evaluation.json could not be read after training"}
    return JSONResponse({"ok": True, "stdout": completed.stdout[-2000:] if completed.stdout else None, "report": report})


@app.get("/api/manifest")
async def api_manifest() -> JSONResponse:
    return JSONResponse(load_manifest())


@app.get("/api/stats")
async def api_stats() -> JSONResponse:
    history = load_history()
    birds = sum(1 for h in history if h.get("decision") == "bird")
    drones = sum(1 for h in history if h.get("decision") == "drone")
    uncertainties = sum(1 for h in history if h.get("decision") == "uncertain")
    errors = sum(1 for h in history if h.get("decision") == "error")
    confidences = [h["confidence"] for h in history if isinstance(h.get("confidence"), (int, float))]
    return JSONResponse({
        "total": len(history),
        "birds": birds,
        "drones": drones,
        "uncertainties": uncertainties,
        "errors": errors,
        "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else None,
        "recent_detections": history[:20],
    })


@app.get("/api/history.json")
async def api_history() -> JSONResponse:
    return JSONResponse(load_history())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
