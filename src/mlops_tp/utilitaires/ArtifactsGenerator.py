import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from joblib import dump as joblib_dump


class ArtifactsGenerator:
    """
    Classe utilitaire pour générer et sauvegarder les artefacts du projet.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        if output_dir is None:
            self.output_dir = Path(__file__).resolve().parent.parent / "artifacts"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: Any, filename: str = "model.joblib") -> Path:
        """Sauvegarde le modèle entraîné au format joblib."""
        path = self.output_dir / filename
        joblib_dump(model, path)
        return path

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json") -> Path:
        """Sauvegarde les métriques d'évaluation."""
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        return path

    def save_feature_schema(
        self, feature_schema: Dict[str, Any], filename: str = "feature_schema.json"
    ) -> Path:
        """Sauvegarde le schéma des variables attendues."""
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(feature_schema, f, indent=4, ensure_ascii=False)
        return path

    def save_run_info(
        self,
        df_train,
        df_val,
        df_test,
        target_column: str,
        train_size,
        val_size,
        test_size,
        random_state,
        dataset_path= None,
        filename: str = "run_info.json",
    ) -> Path:
        """Sauvegarde les informations principales du run d'entraînement."""

        train_total_rows = len(df_train) + len(df_val)
        dataset_total_rows = train_total_rows + len(df_test)

        numeric_features = df_train.drop(columns=[target_column]).select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_features = df_train.drop(columns=[target_column]).select_dtypes(
            include=["object"]
        ).columns.tolist()

        boolean_features = df_train.drop(columns=[target_column]).select_dtypes(
            include=["bool"]
        ).columns.tolist()

        dataset_name = None

        if dataset_path is not None:
            dataset_name = Path(dataset_path).stem
        else:
            dataset_name = "unknown_dataset"

        run_info = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "target_column": target_column,
            "random_state": random_state,
            "columns_names": df_train.columns.tolist(),
            "train_shape": {
                "rows": df_train.shape[0],
                "columns": df_train.shape[1],
            },
            "val_shape": {
                "rows": df_val.shape[0],
                "columns": df_val.shape[1],
            },
            "test_shape": {
                "rows": df_test.shape[0],
                "columns": df_test.shape[1],
            },
            "split": {
                "train_size": float(train_size),
                "val_size": float(val_size),
                "test_size": float(test_size),
                "train_rows": len(df_train),
                "val_rows": len(df_val),
                "test_rows": len(df_test),
                "train_percentage_in_train_val": round(len(df_train) / train_total_rows * 100, 2),
                "val_percentage_in_train_val": round(len(df_val) / train_total_rows * 100, 2),
                "test_percentage_global": round(len(df_test) / dataset_total_rows * 100, 2),
            },
            "features_info": {
                "nombre_total_features": len(df_train.columns) - 1,
                "numerical_features": numeric_features,
                "categorical_features": categorical_features,
                "boolean_features": boolean_features,
            },
            "classes": df_train[target_column].unique().tolist(),
        }

        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=4, ensure_ascii=False)
        return path

    def generate_all(
        self,
        model: Optional[Any] = None,
        metrics: Optional[Dict[str, Any]] = None,
        feature_schema: Optional[Dict[str, Any]] = None,
        run_info: Optional[Dict[str, Any]] = None,
        model_filename: str = "model.joblib",
    ) -> Dict[str, Path]:
        """Génère tous les artefacts fournis."""
        results: Dict[str, Path] = {}

        if model is not None:
            results["model"] = self.save_model(model, filename=model_filename)
        if metrics is not None:
            results["metrics"] = self.save_metrics(metrics)
        if feature_schema is not None:
            results["feature_schema"] = self.save_feature_schema(feature_schema)
        if run_info is not None:
            results["run_info"] = self.save_run_info(**run_info)

        return results