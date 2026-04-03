import json, tempfile, os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
import seaborn as sns
import numpy as np

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
    roc_curve,
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

    # Création du pipeline modèle et paramétrage pour obtenir les meilleurs résultats
    pipeline_model = PipelineModel(
        model_type="gradient_boosting", # on peut modifier le type de model utilise par exemple utilise: "random_forest" ou "logistic_regression"
        n_estimators=100,
        max_depth=10,
        scaler_type="standard",
        numeric_imputer_strategy="mean")
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
    classifier = pipeline.named_steps["classifier"]

    hyperparameters = {
        "classifier": classifier.__class__.__name__,
        "random_state": getattr(classifier, "random_state", None),
    }

    if hasattr(classifier, "n_estimators"):
        hyperparameters["n_estimators"] = classifier.n_estimators

    if hasattr(classifier, "max_depth"):
        hyperparameters["max_depth"] = classifier.max_depth

    metrics = {
        "task_type": "classification",
        "target_column": TARGET_COLUMN,
        "timestamp": datetime.now().isoformat(),
        "train_accuracy": train_accuracy,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "hyperparameters": hyperparameters,
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


    # MLFlow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    mlflow.set_experiment("mon_projet_ml")

    with mlflow.start_run():
        # Paramètres
        classifier = pipeline.named_steps["classifier"]
        mlflow.log_param("model_type", classifier.__class__.__name__)
        mlflow.log_param("random_state",getattr(classifier, "random_state", None))
        if hasattr(classifier, "n_estimators"):
            mlflow.log_param("n_estimators", classifier.n_estimators)

        if hasattr(classifier, "max_depth"):
            mlflow.log_param("max_depth", classifier.max_depth)
        # Métriques
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", accuracy_score(y_val, y_val_pred))
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))
        mlflow.log_metric("test_f1_score", f1_score(y_test, y_test_pred, average="binary", zero_division=0))
        mlflow.log_metric("test_precision", precision_score(y_test, y_test_pred, average="binary", zero_division=0))
        mlflow.log_metric("test_recall", recall_score(y_test, y_test_pred, average="binary", zero_division=0))

        if y_test_proba is not None:
            mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_test_proba))
        print(" the run has started")

        # Artefacts 

        with tempfile.TemporaryDirectory() as tmpdir:
            
            # Matrice de confusion
            cm = np.array(test_metrics["confusion_matrix"])
            labels = sorted(y_test.unique())

            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=labels,yticklabels=labels,linewidths=0.5,linecolor="white",cbar=True,ax=ax_cm,)
            ax_cm.set_title("Matrice de confusion — jeu de test", fontsize=13, pad=12)
            ax_cm.set_xlabel("Classe prédite", fontsize=11)
            ax_cm.set_ylabel("Classe réelle", fontsize=11)
            ax_cm.tick_params(axis="x", rotation=0)
            ax_cm.tick_params(axis="y", rotation=0)
            fig_cm.tight_layout()
            mlflow.log_figure(fig_cm, "confusion_matrix.png")
            plt.close(fig_cm)

            # Courbe ROC
            if y_test_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_test_proba)
                auc_score = roc_auc_score(y_test, y_test_proba)

                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc_score:.3f}")
                ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
                ax.set_xlabel("Taux de faux positifs")
                ax.set_ylabel("Taux de vrais positifs")
                ax.set_title("Courbe ROC")
                ax.legend(loc="lower right")
                mlflow.log_figure(fig, "courbe_roc.png")
                plt.close(fig)

            # Distribution de la cible
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                y_train.value_counts().plot(kind="bar", ax=ax2, color=["steelblue", "coral"])
                ax2.set_title("Distribution de la variable cible (train)")
                ax2.set_xlabel("Classe")
                ax2.set_ylabel("Nombre d'exemples")
                ax2.tick_params(axis="x", rotation=0)
                mlflow.log_figure(fig2, "eda_distribution_cible.png")
                plt.close(fig2)

        # Modèle — premier arg = objet modèle, second = nom du dossier d'artefact (string)
        mlflow.sklearn.log_model(pipeline, "model")
        print("Run MLflow terminé avec succès.")

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