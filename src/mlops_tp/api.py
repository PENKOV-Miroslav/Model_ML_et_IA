import json
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.mlops_tp.inference import InferenceModel


app = FastAPI(
    title="Generic ML Prediction API",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

inference_model = InferenceModel(artifacts_dir=ARTIFACTS_DIR)

with open(ARTIFACTS_DIR / "feature_schema.json", "r", encoding="utf-8") as f:
    feature_schema = json.load(f)

with open(ARTIFACTS_DIR / "metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)


class PredictionInput(BaseModel):
    features: Dict[str, Any] = Field(..., description="Variables d'entrée du modèle")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    return {
        "model_version": "1.0.0",
        "task_type": "classification",
        "feature_schema": feature_schema,
        "validation_accuracy": metrics.get("validation_accuracy"),
        "test_accuracy": metrics.get("test_accuracy"),
    }


@app.post("/predict")
def predict(data: PredictionInput):
    start = time.perf_counter()

    try:
        result = inference_model.predict_with_details(data.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence : {str(e)}")

    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    result["model_version"] = "1.0.0"
    result["latency_ms"] = latency_ms

    return result