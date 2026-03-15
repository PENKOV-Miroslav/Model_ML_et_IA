import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from pathlib import Path

# Ajouter src au path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from mlops_tp.config import (
    RANDOM_STATE, TARGET_COLUMN, TRAIN_PATH, VAL_PATH,
    TEST_PATH, DATA_PATH, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, DROP_COLUMNS)

class Train:

# Cette classe est responsable de la préparation des données pour l'entraînement du modèle. 
# Elle charge les données, effectue la séparation en ensembles d'entraînement et de validation
#, et fournit des informations sur les tailles réelles des ensembles.
# Les chemins d'accès aux données sont définis dans la configuration, et les paramètres de séparation sont également configurables.

    def __init__(self):
        # Initialisation de la classe Train avec les paramètres de configuration
        self.target_column = TARGET_COLUMN
        self.random_state = RANDOM_STATE
        self.train_size = TRAIN_SIZE
        self.val_size = VAL_SIZE
        self.test_size = TEST_SIZE

        # Vérification des ratios
        if abs(self.train_size + self.val_size + self.test_size - 1.0) > 0.001:
            raise ValueError(
                "Train + Val + Test doivent faire 1.0 "
                f"(actuel: {self.train_size + self.val_size + self.test_size})"
            )
        if not 0 < self.train_size < 1:
            raise ValueError("TRAIN_SIZE doit être entre 0 et 1")

        if not 0 < self.val_size < 1:
            raise ValueError("VAL_SIZE doit être entre 0 et 1")

        if not 0 < self.test_size < 1:
            raise ValueError("TEST_SIZE doit être entre 0 et 1")

    def charger_donnee(self):

            # Cas 1 : dataset déjà séparé
            if TRAIN_PATH and TEST_PATH:

                df_train = pd.read_csv(TRAIN_PATH)
                df_test = pd.read_csv(TEST_PATH)

                if VAL_PATH:
                    df_val = pd.read_csv(VAL_PATH)
                else:
                    df_train, df_val = train_test_split(
                        df_train,
                        test_size=self.val_size,
                        stratify=df_train[self.target_column],
                        random_state=self.random_state
                    )

            # Cas 2 : dataset unique
            else:

                df = pd.read_csv(DATA_PATH)

                if DROP_COLUMNS:
                    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

                # Split train+val / test
                df_train_val, df_test = train_test_split(
                    df,
                    test_size=self.test_size,
                    stratify=df[self.target_column],
                    random_state=self.random_state
                )

                # recalcul proportion validation
                val_ratio = self.val_size / (self.train_size + self.val_size)

                df_train, df_val = train_test_split(
                    df_train_val,
                    test_size=val_ratio,
                    stratify=df_train_val[self.target_column],
                    random_state=self.random_state
                )

            return df_train, df_val, df_test

        # -------------------------
        # Séparation features / target
        # -------------------------
    def split_features_target(self, df):
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return X, y