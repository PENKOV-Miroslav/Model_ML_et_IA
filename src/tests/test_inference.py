import pytest
import sys
from pathlib import Path
# Ajouter le dossier src au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))
from mlops_tp.train import Train
from mlops_tp.pipline import PipelineModel

def test_predict_returns_known_classes():
    trainer = Train()
    df_train, df_test = trainer.charger_donnee()
    X_train, _, y_train, _ = trainer.separation_donnee(df_train)
    
    # Pipeline
    pipeline_model = PipelineModel()
    pipeline_model.creer_pipeline(X_train)
    pipeline = pipeline_model.pipeline
    
    # Vérification
    assert pipeline is not None, "Pipeline non créé"
    
    pipeline.fit(X_train, y_train)
    
    # Prédiction sur test
    X_test = df_test.drop(columns=[trainer.target_column])
    y_test_pred = pipeline.predict(X_test)
    
    # Toutes les prédictions doivent être dans les classes connues
    known_classes = df_train[trainer.target_column].unique()
    for pred in y_test_pred:
        assert pred in known_classes, f"Classe inconnue prédite: {pred}"
    
    # Vérification de la taille
    assert len(y_test_pred) == len(X_test), "Nombre de prédictions incorrect"