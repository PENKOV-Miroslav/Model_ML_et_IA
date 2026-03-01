import pandas as pd
import sys
from pathlib import Path

# Ajouter src au path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from src.mlops_tp.train import Train
from src.mlops_tp.utilitaires.ArtifactsGenerator import ArtifactsGenerator

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