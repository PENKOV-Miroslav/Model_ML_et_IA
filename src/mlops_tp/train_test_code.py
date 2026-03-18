import json
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# Ajouter src au path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from mlops_tp.train import Train
from mlops_tp.utilitaires.ArtifactsGenerator import ArtifactsGenerator
from mlops_tp.pipline import PipelineModel
from mlops_tp.config import TARGET_COLUMN, DATA_PATH
from mlops_tp.utilitaires.data_validator import DataValidator


# Fonction pour calculer les métriques du modèle et les retourner dans un format structuré
# pour savoir si le modèle est correct ou non.
def calculer_metriques_classification(y_true, y_pred, y_proba=None):
    """Calcule les métriques principales pour une classification binaire."""
    metriques = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None:
        metriques["roc_auc"] = roc_auc_score(y_true, y_proba)
        metriques["average_precision"] = average_precision_score(y_true, y_proba)

    return metriques


if __name__ == "__main__":

    # Initialisation
    trainer = Train()
    artifacts = ArtifactsGenerator()

    # Chargement et validation des données
    df_train, df_val, df_test = trainer.charger_donnee()
    validator = DataValidator(trainer.target_column)
    validator.validate(df_train)

    # Séparation features / cible
    X_train, y_train = trainer.split_features_target(df_train)
    X_val, y_val = trainer.split_features_target(df_val)
    X_test, y_test = trainer.split_features_target(df_test)

    # Reconstruction des jeux complets
    df_train_final = pd.concat([X_train, y_train], axis=1)
    df_val_final = pd.concat([X_val, y_val], axis=1)

    # Calcul des proportions réelles
    total_rows = len(df_train_final) + len(df_val_final) + len(df_test)
    train_size = len(df_train_final) / total_rows
    val_size = len(df_val_final) / total_rows
    test_size = len(df_test) / total_rows

    # Détection automatique des types de colonnes
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
    boolean_features = X_train.select_dtypes(include=["bool"]).columns.tolist()

    # Création du pipeline
    pipeline_model = PipelineModel()
    pipeline_model.creer_pipeline(X_train)

    assert pipeline_model.pipeline is not None
    pipeline = pipeline_model.pipeline

    # Entraînement
    pipeline.fit(X_train, y_train)

    # Sauvegarde du modèle
    artifacts.save_model(pipeline)

    # Prédictions
    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)

    # Probabilités si disponibles
    y_val_proba = None
    y_test_proba = None

    if hasattr(pipeline, "predict_proba"):
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    # Calcul des métriques
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_metrics = calculer_metriques_classification(y_val, y_val_pred, y_val_proba)
    test_metrics = calculer_metriques_classification(y_test, y_test_pred, y_test_proba)

    # Préparation de metrics.json
    metrics = {
        "task_type": "classification",
        "target_column": TARGET_COLUMN,
        "timestamp": datetime.now().isoformat(),
        "train_accuracy": train_accuracy,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "hyperparameters": {
            "classifier": "RandomForestClassifier",
            "n_estimators": pipeline.named_steps["classifier"].n_estimators,
            "random_state": pipeline.named_steps["classifier"].random_state,
        },
    }

    artifacts.save_metrics(metrics)

    # Génération du schéma des features
    feature_schema = {
        "numerical_features": numeric_features,
        "categorical_features": categorical_features,
        "boolean_features": boolean_features,
        "target_column": TARGET_COLUMN,
    }

    artifacts.save_feature_schema(feature_schema)

    # Sauvegarde des informations de run
    artifacts.save_run_info(
        df_train=df_train_final,
        df_val=df_val_final,
        df_test=df_test,
        target_column=trainer.target_column,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=trainer.random_state,
        dataset_path=DATA_PATH,
    )

    # Affichage console
    print("\n=== Informations sur les données ===")
    print("Features utilisées :", X_train.columns.tolist())
    print("Classes de la cible :", y_train.unique())
    print("Distribution de la cible :")
    print(y_train.value_counts())
    print("Nombre de doublons dans train :", df_train.duplicated().sum())

    print("\n=== Accuracy globale ===")
    print("Accuracy train :", train_accuracy)
    print("Accuracy validation :", accuracy_score(y_val, y_val_pred))
    print("Accuracy test :", accuracy_score(y_test, y_test_pred))

    print("\n=== Métriques validation ===")
    print(json.dumps(val_metrics, indent=2, ensure_ascii=False))

    print("\n=== Métriques test ===")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))

    print("\n=== Tailles des jeux de données ===")
    print("Train size réel :", len(df_train_final))
    print("Validation size réel :", len(df_val_final))
    print("Test size réel :", len(df_test))