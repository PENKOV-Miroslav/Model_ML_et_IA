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


# Réponse TP2

1. Qu’appelle-t-on une expérience dans MLflow?

* Une expérience dans MLflow est le fait de créer un dossier logique qui regroupe toute les tentatives appeler "runs" qui permette de résoudre un même problème afin de comparer ces dernier plus facilement.

2. Qu’appelle-t-on un run?

* Un "run" est une execution individuelle d'un script d'entrainement au sein d'une expèrience,enregistant les configuration du modèle,par exemple les metiques,paramètres,artefacts,tags et information système.

3. Quelle différence faites-vous entre un paramètre, une métrique et un artefact?

* Un paramètre c'est un hyperparamètre d'entrée par exemple: "learning_rate = 0,01" ou "max_depth= 5"
* Une métrique c'est un score mesurer par exemple l'accuracy ou la loss du modèle.
* Un artefact c'est le fichier produit par exemple un csv,graphique.

4. Dans votre propre projet, donnez :
— trois exemples de paramètres que vous pourriez enregistrer;
* RANDOM_STATE
* numeric_features
* categorical_features

— deux ou trois métriques pertinentes selon votre problème;
* "accuracy": 0.9933333333333333
* "f1_score": 0.995260663507109
* "roc_auc": 0.9982847341337907

— un ou deux artefacts utiles à conserver.
*  Le modèle sérialisé (scikit-learn): model.joblib 

5. Elle est accessible sur http://localhost:5050 l'autre adresse dans le tp est inaccessible car protèger
6. On remarque qu'il n' a aucune run ou experience, il y a eu dans mon cas un Default qui a été creer mais a l'interieur il n'y a rien.

13. J'ai choisi de sauvegarder comme paramètres le "random_state", le "model_type" et "n_estimators".

14. "model_type" ("RandomForestClassifier") permet d'identifier immédiatement quel algorithme a été utilisé lors d'un run. C'est essentiel quand on compare plusieurs runs dans l'interface MLflow avec des modèles différents (Random Forest vs Logistic Regression par exemple) — sans ce paramètre, les runs deviennent indiscernables.
"random_state" garantit la reproductibilité. Si deux runs ont exactement les mêmes paramètres mais des random_state différents, les résultats peuvent varier légèrement. L'enregistrer permet de reproduire un run exact à l'identique, ce qui est fondamental en MLOps pour déboguer ou valider un modèle.
"n_estimators" est l'hyperparamètre principal d'un Random Forest : il contrôle le nombre d'arbres construits. C'est le levier le plus direct sur le compromis biais/variance et sur le temps d'entraînement. Le tracker dans MLflow pour tester par exemple 50, 100, 200 arbres et comparer les résultats.

15. Les métriques retenu sont le "train_accuracy","val_accuracy","test_accuracy","test_f1_score","test_precision","test_recall","test_roc_auc"

16. Les métriques couvrent trois jeux de données et plusieurs angles d'évaluation.
"train_accuracy", "val_accuracy" et "test_accuracy" permettent de détecter l'overfitting : si "train_accuracy" est bien supérieure à "test_accuracy", le modèle mémorise les données d'entraînement sans généraliser. Avoir les trois dans MLflow permet de visualiser cet écart directement.
"test_f1_score" est la métrique centrale pour une classification binaire, surtout si les classes sont déséquilibrées. Elle est la moyenne harmonique de la précision et du recall, donc elle pénalise les modèles qui sacrifient l'un pour l'autre.
"test_precision" et test_recall sont enregistrés séparément parce qu'ils mesurent des choses différentes : la précision dit "quand le modèle prédit positif, a-t-il raison ?", le recall dit "parmi tous les vrais positifs, combien a-t-il détectés ?". Selon le contexte métier (détection de fraude, diagnostic médical...), on peut privilégier l'un sur l'autre.
"test_roc_auc" mesure la capacité de discrimination globale du modèle indépendamment du seuil de décision. C'est utile pour comparer des runs même si le seuil change entre eux.

18. J'ai choisi les artefacts,la matrice de confusion,la courbe ROC et la distribution de la variable cible.

19. La matrice de confusion permet de visualiser en détail les erreurs du modèle : combien de faux positifs, faux négatifs, vrais positifs et vrais négatifs. Une accuracy de 90% peut cacher un modèle qui ne détecte jamais la classe minoritaire — la matrice révèle ce que les métriques agrégées masquent.

La courbe ROC montre la capacité de discrimination du modèle sur tous les seuils de décision possibles, pas seulement le seuil par défaut à 0.5. L'aire sous la courbe (AUC) permet de comparer objectivement plusieurs runs dans MLflow indépendamment du seuil choisi.

La distribution de la cible issue de l'EDA est utile pour comprendre le contexte dans lequel les métriques ont été obtenues. Un F1-score de 0.85 n'a pas la même valeur sur des classes équilibrées (50/50) que sur des classes déséquilibrées (90/10).

20. La distribution de la cible est produite avant l'entraînement, lors de l'analyse exploratoire. Elle décrit les données d'entrée.

La matrice de confusion et la courbe ROC sont produites après l'entraînement et l'évaluation, une fois que les prédictions "y_test_pred" et les probabilités "y_test_proba" sont disponibles. Elles décrivent les performances du modèle entraîné.