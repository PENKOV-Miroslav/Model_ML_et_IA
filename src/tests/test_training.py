import pytest
import sys
from pathlib import Path
# Ajouter le dossier src au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))
from mlops_tp.train import Train
from mlops_tp.utilitaires.ArtifactsGenerator import ArtifactsGenerator
from mlops_tp.pipline import PipelineModel
from sklearn.pipeline import Pipeline


def test_training_generates_model(tmp_path):
    # Charger données
    trainer = Train()
    df_train, _ = trainer.charger_donnee()
    X_train, _, y_train, _ = trainer.separation_donnee(df_train)
    
    # Créer pipeline et récupérer le pipeline sklearn
    pipeline_model = PipelineModel()
    pipeline_model.creer_pipeline(X_train)  # <-- indispensable
    pipeline = pipeline_model.pipeline
    
    # Vérification que pipeline n'est pas None
    assert pipeline is not None, "Pipeline non créé"
    assert isinstance(pipeline, Pipeline), "Pipeline n'est pas un objet sklearn Pipeline"
    
    # Entraînement
    pipeline.fit(X_train, y_train)
    
    # Sauvegarde modèle
    artifacts = ArtifactsGenerator(output_dir=tmp_path)
    model_path = artifacts.save_model(pipeline)
    
    # Vérifier que le fichier existe
    assert model_path.exists(), "Le fichier model.joblib n'a pas été généré"