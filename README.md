Licence Apache 2.0

# I - Contexte

Cet ensemble de données contient 6 000 requêtes d'étudiants universitaires étiquetées par niveau de priorité : Élevé, Moyen et Faible. Il est conçu pour les tâches de classification de texte multi-classes dans le traitement du langage naturel (NLP).

L'ensemble de données simule le soutien universitaire réel et les demandes administratives telles que les problèmes d'examen, les retards de bourses d'études, les problèmes de portail, les demandes de renseignements sur les auberges et les questions académiques générales.

Ce dataset est interessant puisque il comporte les exigences du TP, une taille raisonable moins de 200 Mb, des données structurer au format CSV,c'est un problème de classification multi-classe.

# II - La Tâche

Classification

# III - L'antomie des données

* Déséquilibre potentiel des classes entre Élevée, Moyenne et Faible.
* Données textuelles nécessitant un prétraitement (vectorisation TF-IDF ou embeddings).
* Risque de data leakage si le split n’est pas stratifié.
* Hétérogénéité des variables (texte + numérique + catégoriel).

### Types de variables:

* Query_ID – identifiant unique
* Student_Query – Texte de la demande de l’étudiant
* Department – Département universitaire pertinent
* Days_To_Deadline – Temps restant lié à la demande
* Priority_Label – Variable cible (Élevée / Moyenne / Faible)

La taille globale est de 6 000 requêtes d'étudiants universitaires étiquetées par niveau de priorité séparer en deux jeu de donnée "university_query_train"= 5 000 et "university_query_test" = 1 000

# IV - Les défis anticipés
* à completer

# V - Source dataset

* https://www.kaggle.com/datasets/coderanand/university-query-priority-classification?select=university_query_test.csv
* https://www.kaggle.com/datasets/emirhanakku/synthetic-freelance-job-platform-dataset?select=synthetic_freelance_jobs.csv

* https://keylabs.ai/blog/understanding-the-f1-score-and-auc-roc-curve/




# evelment a faire :
 streamlit des donnée analyse univariée,bi-variée du fichier train.csv
 faire le fichier docker container de l'application


 # test /predict
 {
  "features": {
    "category": "Web Development",
    "budget_usd": 1500,
    "duration_days": 30,
    "num_applicants": 12,
    "freelancer_rating": 4.7,
    "completion_time_days": 2
  }
}