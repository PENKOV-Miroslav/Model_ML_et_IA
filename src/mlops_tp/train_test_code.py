import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import sys
from pathlib import Path

# Ajouter src au path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from mlops_tp.train import Train
from mlops_tp.utilitaires.ArtifactsGenerator import ArtifactsGenerator
from mlops_tp.pipline import PipelineModel
from mlops_tp.config import (TARGET_COLUMN)

if __name__ == "__main__":

    # Instancier les classes
    trainer = Train()
    artifacts = ArtifactsGenerator()

    # Charger les données
    df_train, df_test = trainer.charger_donnee()

    # Split train -> train + validation
    X_train, X_val, y_train, y_val = trainer.separation_donnee(df_train)
    Y_validation = y_val.copy()  # Créer une copie de y_val pour la validation
   #y_val = y_val.reset_index(drop=True)  # Réinitialiser les index pour éviter les problèmes de concaténation

    # Reconstruire les DataFrames complets
    df_train_final = pd.concat([X_train, y_train], axis=1)
    df_val = pd.concat([X_val, y_val], axis=1)

    # Calcul des vraies tailles
    total_rows = len(df_train)
    train_size = len(df_train_final) / total_rows
    val_size = len(df_val) / total_rows
    test_size = len(df_test) / (len(df_train) + len(df_test))

    # Définir les colonnes
    numeric_features = ["Days_To_Deadline","Query_ID"]
    categorical_features = ["Department", "Student_Query"]

    # Créer le pipeline via la classe PipelineModel
    pipeline_model = PipelineModel()
    pipeline_model.creer_pipeline(X_train)  # X_train sert à identifier les colonnes
    pipeline = pipeline_model.pipeline

    # Entraîner le pipeline sur le train
    pipeline.fit(X_train, y_train)

    # -------------------------------
    # Sauvegarde du modèle
    # -------------------------------
    artifacts.save_model(pipeline)

    # -------------------------------
    # Prédiction et métriques
    # -------------------------------
    # Validation
    y_val_pred = pipeline.predict(X_val)
    val_accuracy = accuracy_score(Y_validation, y_val_pred)
    val_f1 = f1_score(Y_validation, y_val_pred, average='weighted')

    # Test
    X_test = df_test.drop(columns=[trainer.target_column])
    y_test = df_test[trainer.target_column]
    y_test_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # -------------------------------
    # Préparer metrics.json complet
    # -------------------------------
    metrics = {
        "validation_accuracy": val_accuracy,
        "validation_f1_score": val_f1,
        "test_accuracy": test_accuracy,
        "test_f1_score": test_f1,
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "classifier": "RandomForestClassifier",
            "n_estimators": pipeline.named_steps['classifier'].n_estimators,
            "random_state": pipeline.named_steps['classifier'].random_state
        }
    }

    # Sauvegarder metrics.json
    artifacts.save_metrics(metrics)

    # -------------------------------
    # Génération du feature_schema.json
    # -------------------------------

    feature_schema = {
        "numerical_features": numeric_features,
        "categorical_features": categorical_features,
        "target_column": TARGET_COLUMN
    }

    # Sauvegarder feature_schema.json
    artifacts.save_feature_schema(feature_schema)

    # -------------------------------
    # Afficher les métriques
    # -------------------------------
    print("Métriques Validation:", val_accuracy, val_f1)
    print("Métriques Test:", test_accuracy, test_f1)

    print("Train size réel:", len(df_train_final))
    print("Validation size réel:", len(df_val))
    print("Test size réel:", len(df_test))

    # Sauvegarde des infos de run
    artifacts.save_run_info(
        df_train=df_train_final,
        df_val=df_val,
        df_test=df_test,
        target_column=trainer.target_column,   # récupéré depuis config
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=trainer.random_state,     # récupéré depuis config
    )