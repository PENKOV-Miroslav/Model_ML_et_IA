import sys
from pathlib import Path

import pandas as pd

# Ajouter src au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

from mlops_tp.train import Train
from mlops_tp.pipline import PipelineModel
from mlops_tp.utilitaires.ArtifactsGenerator import ArtifactsGenerator


def test_training_pipeline_runs_and_saves_model(tmp_path):
    trainer = Train()
    artifacts = ArtifactsGenerator(output_dir=tmp_path)

    df_train, df_val, df_test = trainer.charger_donnee()

    assert not df_train.empty
    assert not df_val.empty
    assert not df_test.empty
    assert trainer.target_column in df_train.columns

    X_train, y_train = trainer.split_features_target(df_train)
    X_val, y_val = trainer.split_features_target(df_val)

    assert not X_train.empty
    assert not y_train.empty

    pipeline_model = PipelineModel()
    pipeline_model.creer_pipeline(X_train)

    assert pipeline_model.pipeline is not None
    pipeline = pipeline_model.pipeline

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    assert len(y_pred) == len(y_val)

    model_path = artifacts.save_model(pipeline)

    assert model_path.exists()
    assert model_path.name == "model.joblib"


def test_training_generates_all_main_artifacts(tmp_path):
    trainer = Train()
    artifacts = ArtifactsGenerator(output_dir=tmp_path)

    df_train, df_val, df_test = trainer.charger_donnee()

    X_train, y_train = trainer.split_features_target(df_train)
    X_val, y_val = trainer.split_features_target(df_val)

    pipeline_model = PipelineModel()
    pipeline_model.creer_pipeline(X_train)

    assert pipeline_model.pipeline is not None
    pipeline = pipeline_model.pipeline

    pipeline.fit(X_train, y_train)

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
    boolean_features = X_train.select_dtypes(include=["bool"]).columns.tolist()

    metrics = {
        "validation_accuracy": 0.0,
        "validation_f1_score": 0.0,
        "test_accuracy": 0.0,
        "test_f1_score": 0.0,
    }

    feature_schema = {
        "numerical_features": numeric_features,
        "categorical_features": categorical_features,
        "boolean_features": boolean_features,
        "target_column": trainer.target_column,
    }

    df_train_final = pd.concat([X_train, y_train], axis=1)
    df_val_final = pd.concat([X_val, y_val], axis=1)

    total_rows = len(df_train_final) + len(df_val_final) + len(df_test)
    train_size = len(df_train_final) / total_rows
    val_size = len(df_val_final) / total_rows
    test_size = len(df_test) / total_rows

    artifacts.save_model(pipeline)
    metrics_path = artifacts.save_metrics(metrics)
    schema_path = artifacts.save_feature_schema(feature_schema)
    run_info_path = artifacts.save_run_info(
        df_train=df_train_final,
        df_val=df_val_final,
        df_test=df_test,
        target_column=trainer.target_column,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=trainer.random_state,
    )

    assert (tmp_path / "model.joblib").exists()
    assert metrics_path.exists()
    assert schema_path.exists()
    assert run_info_path.exists()