import sys
import pandas as pd
import sklearn as sk
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ajoute le dossier src/ au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))
from mlops_tp.utilitaires.ArtifactsGenerator import ArtifactsGenerator


# Crée le générateur d’artefacts
artifacts = ArtifactsGenerator()

etat_aléatoire = 42
target_column = "Priority_Label"

def charger_donnee():
    df_train = pd.read_csv("data/university_query_train.csv")
    df_test = pd.read_csv("data/university_query_test.csv")
    return df_train, df_test

# sépare les données d’entraînement en features (X_train) et labels/cible (y_train)
def separation_donnee(df_train):
    X_train, X_val, y_train, y_val = train_test_split(
        df_train.drop(columns=[target_column]),
        df_train[target_column],
        test_size=0.15,  # 15% pour la validation
        stratify=df_train[target_column],
        random_state=etat_aléatoire
    )
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    df_train, df_test = charger_donnee()

    # Split train -> train + validation
    X_train, X_val, y_train, y_val = separation_donnee(df_train)

    # Reconstruire les DataFrames complets
    df_train_final = pd.concat([X_train, y_train], axis=1)
    df_val = pd.concat([X_val, y_val], axis=1)

    # Calcul des vraies tailles
    total_rows = len(df_train)
    train_size = len(df_train_final) / total_rows
    val_size = len(df_val) / total_rows
    test_size = len(df_test) / (len(df_train) + len(df_test))

    print("Train size réel:", len(df_train_final))
    print("Validation size réel:", len(df_val))
    print("Test size réel:", len(df_test))

    artifacts.save_run_info(
        df_train=df_train_final,
        df_val=df_val,
        df_test=df_test,
        target_column=target_column,
        train_size=train_size,   # ✅ float
        val_size=val_size,       # ✅ float
        test_size=test_size,     # ✅ float
        random_state=etat_aléatoire,
    )