import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd


class InferenceModel:
    def __init__(self, artifacts_dir: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent

        if artifacts_dir is None:
            artifacts_dir = base_dir / "artifacts"

        self.artifacts_dir = Path(artifacts_dir)
        self.model_path = self.artifacts_dir / "model.joblib"
        self.schema_path = self.artifacts_dir / "feature_schema.json"
        self.metrics_path = self.artifacts_dir / "metrics.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {self.model_path}")

        self.model = joblib.load(self.model_path)

        self.feature_schema = {}
        if self.schema_path.exists():
            with open(self.schema_path, "r", encoding="utf-8") as f:
                self.feature_schema = json.load(f)

        self.expected_features = (
            self.feature_schema.get("numerical_features", [])
            + self.feature_schema.get("categorical_features", [])
            + self.feature_schema.get("boolean_features", [])
        )

    def validate_input(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(features, dict):
            raise ValueError("Les features doivent être fournies sous forme de dictionnaire.")

        missing_features = [col for col in self.expected_features if col not in features]

        if missing_features:
            raise ValueError(
                f"Variables manquantes pour l'inférence : {missing_features}"
            )

        # Ignore les colonnes en trop et remet les colonnes dans l'ordre attendu
        ordered_features = {col: features[col] for col in self.expected_features}
        return ordered_features

    def predict(self, features: Dict[str, Any]) -> Any:
        ordered_features = self.validate_input(features)
        input_df = pd.DataFrame([ordered_features])
        prediction = self.model.predict(input_df)[0]

        if hasattr(prediction, "item"):
            return prediction.item()
        return prediction

    def predict_proba(self, features: Dict[str, Any]) -> Optional[Dict[str, float]]:
        if not hasattr(self.model, "predict_proba"):
            return None

        ordered_features = self.validate_input(features)
        input_df = pd.DataFrame([ordered_features])

        probabilities = self.model.predict_proba(input_df)[0]
        classes = self.model.classes_

        return {
            str(cls): float(prob)
            for cls, prob in zip(classes, probabilities)
        }

    def predict_with_details(self, features: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "prediction": self.predict(features),
            "task": "classification",
        }

        proba = self.predict_proba(features)
        if proba is not None:
            result["proba"] = proba

        return result