import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ajouter src au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

from mlops_tp.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert isinstance(data["status"], str)
    assert "ok" in data["status"].lower()


def test_metadata(client):
    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()
    assert "model_version" in data
    assert "task_type" in data
    assert "validation_accuracy" in data
    assert "test_accuracy" in data


def test_predict_valid(client):
    payload = {
        "features": {
            "category": "Web Development",
            "budget_usd": 1200,
            "duration_days": 15,
            "num_applicants": 8,
            "freelancer_rating": 4.6,
            "completion_time_days": 14
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "task" in data
    assert "model_version" in data


def test_predict_missing_feature(client):
    payload = {
        "features": {
            "category": "Web Development",
            "budget_usd": 1200,
            "duration_days": 15,
            "num_applicants": 8,
            "freelancer_rating": 4.6
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_extra_feature(client):
    payload = {
        "features": {
            "category": "Web Development",
            "budget_usd": 1200,
            "duration_days": 15,
            "num_applicants": 8,
            "freelancer_rating": 4.6,
            "completion_time_days": 14,
            "unknown_feature": "test"
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code in [200, 422]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "features": {
                "category": "Web Development",
                "budget_usd": 800,
                "duration_days": 10,
                "num_applicants": 5,
                "freelancer_rating": 4.5,
                "completion_time_days": 9
            }
        },
        {
            "features": {
                "category": "Design",
                "budget_usd": 1500,
                "duration_days": 20,
                "num_applicants": 10,
                "freelancer_rating": 4.9,
                "completion_time_days": 18
            }
        }
    ]
)
def test_predict_multiple_inputs(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200