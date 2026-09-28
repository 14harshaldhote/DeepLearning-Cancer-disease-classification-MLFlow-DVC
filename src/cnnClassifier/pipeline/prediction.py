import base64
import io
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from cnnClassifier.components.explainability import GradCAM, overlay
from cnnClassifier.constants import CLASS_LABELS, CLASS_NAMES

IMAGE_SIZE = (224, 224)


def load_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def preprocess(image: Image.Image) -> np.ndarray:
    """Same preprocessing as training: RGB, 224x224 bilinear, scaled to [0, 1]."""
    resized = image.resize(IMAGE_SIZE, Image.BILINEAR)
    return (np.asarray(resized, dtype=np.float32) / 255.0)[None, ...]


def to_png_base64(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class PredictionPipeline:
    """Loads the model once and serves predictions with an explanation heatmap."""

    def __init__(self, model_path: str | Path = "model/model.h5", review_threshold: float = 0.8):
        self.model_path = Path(model_path)
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.gradcam = GradCAM(self.model)
        self.review_threshold = review_threshold

    def predict(self, data: bytes, explain: bool = True) -> dict:
        start = time.perf_counter()
        image = load_image(data)
        batch = preprocess(image)
        probs = self.model(batch, training=False).numpy()[0]
        index = int(probs.argmax())
        confidence = float(probs[index])

        result = {
            "class": CLASS_NAMES[index],
            "label": CLASS_LABELS[CLASS_NAMES[index]],
            "confidence": confidence,
            "probabilities": {CLASS_LABELS[n]: float(p) for n, p in zip(CLASS_NAMES, probs, strict=True)},
            "needs_review": confidence < self.review_threshold,
            "review_threshold": self.review_threshold,
        }

        if explain:
            heatmap = self.gradcam.heatmap(batch, index)
            display = np.asarray(image.resize((448, 448), Image.BILINEAR))
            result["heatmap_png"] = to_png_base64(overlay(display, heatmap))

        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return result
