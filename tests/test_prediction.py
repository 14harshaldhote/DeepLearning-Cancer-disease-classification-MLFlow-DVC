import numpy as np
import pytest
from PIL import Image

from cnnClassifier.components.explainability import jet_colormap, overlay
from cnnClassifier.pipeline.prediction import PredictionPipeline, preprocess


@pytest.fixture(scope="module")
def pipeline(root):
    return PredictionPipeline(root / "model" / "model.h5", review_threshold=0.8)


def test_preprocess_matches_training_scale():
    image = Image.new("RGB", (512, 400), color=(255, 128, 0))
    batch = preprocess(image)
    assert batch.shape == (1, 224, 224, 3)
    assert batch.max() <= 1.0 and batch.min() >= 0.0
    assert batch[0, 0, 0, 0] == pytest.approx(1.0)


def test_prediction_output_shape(pipeline, samples):
    result = pipeline.predict(samples[0].read_bytes())
    assert result["class"] in {"adenocarcinoma", "normal"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert 0.5 <= result["confidence"] <= 1.0
    assert result["needs_review"] == (result["confidence"] < 0.8)
    assert result["heatmap_png"]


def test_bundled_samples_are_classified_correctly(pipeline, samples):
    """Smoke test on held-out images: catches preprocessing regressions such as a missing /255."""
    correct = sum(
        pipeline.predict(s.read_bytes(), explain=False)["class"] == s.stem.rsplit("_", 1)[0]
        for s in samples
    )
    assert correct >= len(samples) - 1


def test_overlay_keeps_image_size():
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    heat = np.random.rand(14, 14).astype(np.float32)
    assert overlay(image, heat).shape == (50, 60, 3)
    assert jet_colormap(np.array([0.0, 1.0])).shape == (2, 3)
