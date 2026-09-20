"""
Tests del servicio de predicción.

Requieren el modelo campeón exportado; si no existe, se omiten:
    python -m flows.training_flow
"""
import pytest

from src.api.schemas import CLAIM_EXAMPLE
from src.config import CHAMPION_EXPORT_DIR

pytestmark = pytest.mark.skipif(
    not (CHAMPION_EXPORT_DIR / "MLmodel").exists(),
    reason="No hay modelo campeón exportado en models/champion/",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_model_info_exposes_version_and_threshold(client):
    body = client.get("/model-info").json()
    assert body["version"] >= 1
    assert 0 < body["decision_threshold"] < 1


def test_predict_returns_probability_and_decision(client):
    body = client.post("/predict", json=CLAIM_EXAMPLE).json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)


def test_predict_rejects_invalid_category(client):
    payload = {**CLAIM_EXAMPLE, "Fault": "Nadie"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_missing_field(client):
    payload = {key: value for key, value in CLAIM_EXAMPLE.items() if key != "Age"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_batch_scores_every_claim(client):
    response = client.post("/predict/batch", json={"claims": [CLAIM_EXAMPLE] * 3})
    body = response.json()
    assert body["n_claims"] == 3
    assert len(body["predictions"]) == 3
