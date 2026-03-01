import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from joblib import dump as joblib_dump



class ArtifactsGenerator:
            
    """
        Classe utilitaire pour générer et sauvegarder des artefacts de run.
                    Méthodes principales:
                        - save_model: sauvegarde un objet modèle au format joblib (`model.joblib`).
                        - save_metrics: écrit un dictionnaire de métriques dans `metrics.json`.
                        - save_feature_schema: écrit un schéma de features dans `feature_schema.json`.
                        - save_run_info: écrit les informations d'exécution dans `run_info.json` (utilise `gen_run_info_json`).
                        - generate_all: exécute toutes les sauvegardes demandées selon les paramètres fournis.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialise l'instance.

        Si `output_dir` n'est pas fourni, utilise le dossier `artifacts`
        au niveau du package `src/mlops_tp/artifacts`.
        """
        if output_dir is None:
            # fichier actuele: src/mlops_tp/utilitaires/ArtifactsGenerator.py
            # target dossier artifacts: src/mlops_tp/artifacts
            default_artifacts = Path(__file__).resolve().parent.parent / 'artifacts'
            self.output_dir = default_artifacts
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: Any, filename: str = 'model.joblib') -> Path:
        """Sauvegarde le modèle avec joblib.

        Raises:
            RuntimeError: si `joblib` n'est pas disponible.
        """
        if joblib_dump is None:
            raise RuntimeError('joblib est indisponible. Installer joblib pour effectuer la sauvegarde du modèle.')
        path = self.output_dir / filename
        joblib_dump(model, path)
        return path

    def save_metrics(self, metrics: Dict[str, Any], filename: str = 'metrics.json') -> Path:
        path = self.output_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        return path

    def save_feature_schema(self, feature_schema: Dict[str, Any], filename: str = 'feature_schema.json') -> Path:
        path = self.output_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(feature_schema, f, indent=4, ensure_ascii=False)
        return path

    def save_run_info(self, run_info: Dict[str, Any], filename: str = 'run_info.json') -> Path:
        """Écrit les informations d'exécution dans `run_info.json` et ajoute un timestamp."""
        run_info = dict(run_info) if run_info is not None else {}
        run_info['timestamp'] = datetime.now().isoformat()
        path = self.output_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(run_info, f, indent=4, ensure_ascii=False)
        return path

    def generate_all(self,
                     model: Optional[Any] = None,
                     metrics: Optional[Dict[str, Any]] = None,
                     feature_schema: Optional[Dict[str, Any]] = None,
                     run_info: Optional[Dict[str, Any]] = None,
                     model_filename: str = 'model.joblib') -> Dict[str, Path]:
        """Génère tous les artefacts fournis et retourne les chemins créés.

        Seuls les éléments non-`None` seront sauvegardés.
        """
        results: Dict[str, Path] = {}
        if model is not None:
            results['model'] = self.save_model(model, filename=model_filename)
        if metrics is not None:
            results['metrics'] = self.save_metrics(metrics)
        if feature_schema is not None:
            results['feature_schema'] = self.save_feature_schema(feature_schema)
        if run_info is not None:
            results['run_info'] = self.save_run_info(run_info)
        return results
