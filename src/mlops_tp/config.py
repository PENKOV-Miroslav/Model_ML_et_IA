# src/mlops_tp/config.py
# Ce fichier contient les configurations globales pour le projet MLOps TP

# 
RANDOM_STATE = 42

# Nom de la colonne cible (label) dans les données d'entraînement
TARGET_COLUMN = "hired" # Priority_Label

# Chemins vers les données d'entraînement et de test (à adapter selon le besoin)
#TRAIN_PATH = "data/university_query_train.csv"
#TEST_PATH = "data/university_query_test.csv"
#VAL_PATH ="0.15"
TRAIN_PATH = None
VAL_PATH = None
TEST_PATH = None

DATA_PATH = "data/synthetic_freelance_jobs.csv"

 # Proportion de données à utiliser pour l'entraînement (ex: 0.7 pour 70%)
TRAIN_SIZE = 0.7

 # Proportion de données à utiliser pour la validation (ex: 0.15 pour 15%)
VAL_SIZE = 0.15

 # Proportion de données à utiliser pour le test (ex: 0.15 pour 15%)
TEST_SIZE = 0.15

# colonnes optionnelles à ignorer (ex: id = DROP_COLUMNS = ["job_id"])
DROP_COLUMNS = [
    "job_id",
    "job_title",
    "job_description",
    "posted_date",
    "success"
]