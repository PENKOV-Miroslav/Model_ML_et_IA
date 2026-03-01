import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.model_selection import train_test_split

etat_aléatoire = 42
target_column = "Priority_Label"

def charger_donnee():
    df_train = pd.read_csv("data/university_query_train.csv")
    df_test = pd.read_csv("data/university_query_test.csv")
    return df_train, df_test

# sépare les données d’entraînement en features (X_train) et labels/cible (y_train)
def séparation_donnee(df_train,target_column):
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    return X_train, y_train

def séparation_donnée_entrainement_validation(df_train, target_column):
    X_train, X_val, y_train, y_val = train_test_split(
        df_train.drop(columns=[target_column]),
        df_train[target_column],
        test_size=0.15, # 15% pour la validation
        stratify=df_train[target_column],
        random_state= etat_aléatoire # pour la reproductibilité
    )
    return X_train, X_val, y_train, y_val

