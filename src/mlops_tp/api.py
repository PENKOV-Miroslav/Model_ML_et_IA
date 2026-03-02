# src/mlops_tp/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
from pathlib import Path
from typing import Dict, Any

# ----------------------------------
# Initialisation API
# ----------------------------------
app = FastAPI(
    title="University Query Priority API",
    version="1.0"
)

# ----------------------------------
# Charger artefacts au démarrage
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

model = joblib.load(ARTIFACTS_DIR / "model.joblib")

with open(ARTIFACTS_DIR / "feature_schema.json", "r") as f:
    feature_schema = json.load(f)

with open(ARTIFACTS_DIR / "metrics.json", "r") as f:
    metrics = json.load(f)


# ----------------------------------
# Modèle d'entrée pour /predict
# ----------------------------------
class PredictionInput(BaseModel):
    Query_ID: int
    Student_Query: str
    Department: str
    Days_To_Deadline: int


# ----------------------------------
# 1️⃣ Health Check
# ----------------------------------
@app.get("/health")
def health():
    return {"status": "API is running"}


# ----------------------------------
# 2️⃣ Metadata
# ----------------------------------
@app.get("/metadata")
def metadata():
    return {
        "model_version": "1.0",
        "task_type": "classification",
        "features": feature_schema,
        "validation_accuracy": metrics.get("validation_accuracy"),
        "test_accuracy": metrics.get("test_accuracy")
    }


# ----------------------------------
# 3️⃣ Prediction
# ----------------------------------
@app.post("/predict")
def predict(data: PredictionInput):
    # Convertir en dictionnaire
    input_dict = data.model_dump()

    # Faire la prédiction (mettre sous forme de liste pour sklearn)
    prediction = model.predict([input_dict])[0]

    return {
        "input": input_dict,
        "prediction": prediction
    }