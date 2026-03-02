import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from pathlib import Path

# Ajouter src au path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from mlops_tp.config import (RANDOM_STATE, TARGET_COLUMN, TRAIN_PATH, TEST_PATH, VAL_SIZE,)

class Train:

# Cette classe est responsable de la préparation des données pour l'entraînement du modèle. 
# Elle charge les données, effectue la séparation en ensembles d'entraînement et de validation
#, et fournit des informations sur les tailles réelles des ensembles.
# Les chemins d'accès aux données sont définis dans la configuration, et les paramètres de séparation sont également configurables.

    def __init__(self):
        # Initialisation de la classe Train avec les paramètres de configuration
        self.target_column = TARGET_COLUMN
        self.random_state = RANDOM_STATE
        self.val_size = VAL_SIZE

    def charger_donnee(self):
        # Charger les données d'entraînement et de test à partir des chemins spécifiés dans la configuration
        df_train = pd.read_csv(TRAIN_PATH)
        df_test = pd.read_csv(TEST_PATH)
        return df_train, df_test

    def separation_donnee(self, df_train):
        # Séparation des données en ensembles d'entraînement et de validation
        X_train, X_val, y_train, y_val = train_test_split(
            df_train.drop(columns=[self.target_column]), # Utiliser drop pour séparer les features de la target
            df_train[self.target_column], # Utiliser la colonne cible pour y_val
            test_size=self.val_size, # Utiliser la taille de validation définie dans la configuration
            stratify=df_train[self.target_column], # Assurer que la répartition des classes est similaire dans les ensembles d'entraînement et de validation
            random_state=self.random_state # Utiliser le random_state de la configuration pour assurer la reproductibilité
        )
        return X_train, X_val, y_train, y_val