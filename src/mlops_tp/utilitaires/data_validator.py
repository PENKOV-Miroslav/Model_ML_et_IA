import pandas as pd


class DataValidator:

    def __init__(self, target_column):
        self.target_column = target_column

    def validate(self, df):

        # 1 Vérifier si la target existe
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' manquante"
            )

        # 2 Dataset vide
        if df.empty:
            raise ValueError("Dataset vide")

        # 3 Vérifier colonnes constantes
        constant_cols = [
            col for col in df.columns
            if df[col].nunique() == 1
        ]

        if constant_cols:
            print("Colonnes constantes :", constant_cols)

        # 4 Vérifier NaN
        nan_ratio = df.isna().mean()

        high_nan = nan_ratio[nan_ratio > 0.5]

        if not high_nan.empty:
            print("Colonnes avec >50% NaN :", list(high_nan.index))

        # 5 Vérifier doublons
        duplicates = df.duplicated().sum()

        if duplicates > 0:
            print("Lignes dupliquées :", duplicates)

        print("Validation dataset OK")