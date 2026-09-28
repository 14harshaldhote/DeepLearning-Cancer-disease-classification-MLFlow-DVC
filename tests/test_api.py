import base64

import pytest
from fastapi.testclient import TestClient

import app as app_module


@pytest.fixture(scope="module")
def client():
    with TestClient(app_module.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_dashboard_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Chest CT Classifier" in res.text


def test_model_info(client):
    body = client.get("/api/model-info").json()
    assert body["params"]["CLASSES"] == 2
    assert body["evaluation"]["n_samples"] > 0


def test_predict_upload(client, samples):
    with samples[0].open("rb") as f:
        res = client.post("/api/predict", files={"file": (samples[0].name, f, "image/png")})
    assert res.status_code == 200
    body = res.json()
    assert {"label", "confidence", "probabilities", "needs_review", "heatmap_png"} <= body.keys()
    assert client.get("/api/history").json()[0]["source"] == samples[0].name


def test_predict_rejects_non_image(client):
    res = client.post("/api/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400


def test_legacy_endpoint(client, samples):
    encoded = base64.b64encode(samples[-1].read_bytes()).decode()
    res = client.post("/predict", json={"image": encoded})
    assert res.status_code == 200
    assert res.json()[0]["image"] in {"Normal", "Adenocarcinoma Cancer"}


def test_samples_listed(client):
    samples = client.get("/api/samples").json()
    assert {s["true_class"] for s in samples} == {"adenocarcinoma", "normal"}
