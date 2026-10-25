"""API contract: happy path, validation rejections, missing artifact."""

import dataclasses
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from production_app import model_service
from production_app.api import app
from production_app.config import settings
from production_app.model_service import load_model
from production_app.schemas import FEATURES

client = TestClient(app)


@pytest.fixture
def payload() -> dict[str, float]:
    row = pd.read_csv(settings.data_path).iloc[0]
    return {name: float(row[name]) for name in FEATURES}


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_valid(payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("benign", "malignant")


@pytest.mark.parametrize(
    "change",
    [
        lambda p: {k: v for k, v in p.items() if k != FEATURES[0]},
        lambda p: {**p, "unexpected_feature": 1.0},
        lambda p: {**p, FEATURES[0]: -1.0},
        lambda p: {**p, FEATURES[0]: float("nan")},
    ],
    ids=["missing", "extra", "negative", "non-finite"],
)
def test_predict_rejects_invalid_input(payload, change):
    # json.dumps keeps NaN literal so the payload reaches the validator
    response = client.post(
        "/predict",
        content=json.dumps(change(payload)),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


@pytest.fixture
def missing_artifact(monkeypatch, tmp_path):
    load_model.cache_clear()
    monkeypatch.setattr(
        model_service,
        "settings",
        dataclasses.replace(
            model_service.settings, model_path=tmp_path / "missing.joblib"
        ),
    )
    yield
    load_model.cache_clear()


def test_model_missing_returns_503(missing_artifact, payload):
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json=payload).status_code == 503
