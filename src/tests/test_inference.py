import sys
from pathlib import Path

import pytest

# Ajouter src au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

from mlops_tp.inference import InferenceModel


@pytest.fixture
def inference_model():
    return InferenceModel()


@pytest.fixture
def valid_features():
    return {
        "category": "Web Development",
        "budget_usd": 1200,
        "duration_days": 15,
        "num_applicants": 8,
        "freelancer_rating": 4.6,
        "completion_time_days": 14
    }


def test_predict_returns_valid_class(inference_model, valid_features):
    prediction = inference_model.predict(valid_features)

    assert isinstance(prediction, (bool, str, int))


def test_predict_proba_returns_values_between_0_and_1(inference_model, valid_features):
    proba = inference_model.predict_proba(valid_features)

    if proba is None:
        pytest.skip("Le modèle ne supporte pas predict_proba.")

    assert isinstance(proba, dict)
    assert len(proba) > 0

    for _, value in proba.items():
        assert 0.0 <= value <= 1.0


def test_predict_with_missing_feature_raises_error(inference_model):
    invalid_features = {
        "category": "Web Development",
        "budget_usd": 1200,
        "duration_days": 15,
        "num_applicants": 8,
        "freelancer_rating": 4.6
    }

    with pytest.raises(ValueError):
        inference_model.predict(invalid_features)


def test_predict_with_details_contains_prediction(inference_model, valid_features):
    result = inference_model.predict_with_details(valid_features)

    assert "prediction" in result
    assert "task" in result
    assert result["task"] == "classification"

    if "proba" in result:
        assert isinstance(result["proba"], dict)