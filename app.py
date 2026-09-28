"""FastAPI service: prediction API with Grad-CAM explanations plus the web dashboard.

Run locally:  uvicorn app:app --port 8080
API docs:     http://localhost:8080/docs
"""

import base64
import binascii
import json
import os
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError
from pydantic import BaseModel

from cnnClassifier.pipeline.prediction import PredictionPipeline

ROOT = Path(__file__).parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "model" / "model.h5"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())

state: dict = {"history": deque(maxlen=25)}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the model once at startup instead of on every request.
    state["pipeline"] = PredictionPipeline(MODEL_PATH, PARAMS["REVIEW_THRESHOLD"])
    yield
    state.clear()


app = FastAPI(
    title="Chest CT Cancer Classifier",
    description="VGG16 transfer-learning model that classifies chest CT slices as "
    "adenocarcinoma or normal, with Grad-CAM explanations. Research demo, not a medical device.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def run_prediction(data: bytes, source: str, explain: bool = True) -> dict:
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image larger than 10 MB")
    try:
        result = state["pipeline"].predict(data, explain=explain)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Not a readable image (use PNG or JPEG)")

    state["history"].appendleft(
        {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "source": source,
            "label": result["label"],
            "confidence": round(result["confidence"], 4),
            "needs_review": result["needs_review"],
            "latency_ms": result["latency_ms"],
        }
    )
    return result


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(ROOT / "templates" / "index.html")


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "model_loaded": "pipeline" in state, "model_path": str(MODEL_PATH)}


@app.get("/api/model-info", tags=["model"])
def model_info():
    """Everything the dashboard shows about the model: config, data split, training, test metrics."""
    return {
        "architecture": "VGG16 (ImageNet weights, frozen) + global average pooling, dropout, softmax head",
        "input_size": PARAMS["IMAGE_SIZE"],
        "params": PARAMS,
        "scores": read_json(ROOT / "scores.json"),
        "evaluation": read_json(ROOT / "reports" / "evaluation.json"),
        "training_history": read_json(ROOT / "reports" / "training_history.json"),
        "data_split": read_json(ROOT / "reports" / "data_split.json"),
    }


@app.get("/api/samples", tags=["model"])
def samples():
    """Held-out test images bundled with the app so visitors can try it without their own scans."""
    files = sorted((ROOT / "static" / "samples").glob("*.png"))
    return [
        {"name": f.stem, "true_class": f.stem.rsplit("_", 1)[0], "url": f"/static/samples/{f.name}"}
        for f in files
    ]


@app.post("/api/predict", tags=["prediction"])
def predict(
    file: UploadFile = File(..., description="Chest CT slice, PNG or JPEG"),
    explain: bool = Query(True, description="Include a Grad-CAM heatmap (base64 PNG)"),
):
    return run_prediction(file.file.read(), source=file.filename or "upload", explain=explain)


@app.get("/api/history", tags=["prediction"])
def history():
    """Most recent predictions (no images are stored) for the dashboard's audit trail."""
    return list(state["history"])


class LegacyRequest(BaseModel):
    image: str


@app.post("/predict", tags=["prediction"], deprecated=True)
def predict_legacy(body: LegacyRequest):
    """Old endpoint kept for existing clients: base64 image in, [{"image": label}] out."""
    try:
        data = base64.b64decode(body.image)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 image")
    result = run_prediction(data, source="legacy-api", explain=False)
    return [{"image": result["label"]}]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
