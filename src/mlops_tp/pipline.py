from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

from mlops_tp.train import Train


class PipelineModel:

    # Classe chargée de construire le pipeline complet de prétraitement + modèle.
    # L'objectif est d'avoir une structure unique, réutilisable et paramétrable.

    # définition des hyperparamétres
    def __init__(
        self,
        model_type="random_forest", #logistic_regression
        random_state=None, # par défaut de la config = 42
        n_estimators=100, # 200
        max_depth=None, # 10
        scaler_type="standard", # minmax
        numeric_imputer_strategy="median", #mean
        categorical_imputer_strategy="most_frequent",
    ):
        # Récupération de la configuration générale du projet
        self.trainer = Train()

        # Le pipeline final sera stocké ici après sa création
        self.pipeline = None

        # Paramètres du modèle et du prétraitement
        self.model_type = model_type
        self.random_state = (
            random_state if random_state is not None else self.trainer.random_state
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.scaler_type = scaler_type
        self.numeric_imputer_strategy = numeric_imputer_strategy
        self.categorical_imputer_strategy = categorical_imputer_strategy

    def construire_scaler(self):
        # Permet de choisir dynamiquement le type de normalisation
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        # StandardScaler est utilisé par défaut
        return StandardScaler()

    def construire_model(self):
        # Construction d'une régression logistique
        if self.model_type == "logistic_regression":
            return LogisticRegression(
                random_state=self.random_state,
                solver="lbfgs", #'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'
                max_iter=1000
            )
        
            # Construction d'un Arbre de décision
        if self.model_type == "decision_tree":
            return DecisionTreeClassifier(
                random_state=self.random_state,
                max_leaf_nodes= 10,
                max_depth=self.max_depth
            )
        
        # Construction d'une forêt aléatoire   
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth
            )
        
        # Construction d'un Extra Trees
        if self.model_type == "extra_trees":
            return ExtraTreesClassifier(
                random_state=self.random_state,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                criterion="gini", #Literal['gini', 'entropy', 'log_loss']
            )

        # Construction d'un Gradient Boosting
        if self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=self.n_estimators,
                learning_rate= 0.1, #Float = 0.1,0.2 etc...
            )
        
        # Sécurité si un type de modèle non prévu est passé en paramètre
        raise ValueError(f"model_type non supporté : {self.model_type}")

    def creer_pipeline(self, X_train):
        # Identification automatique des types de variables à partir du jeu d'entraînement
        numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns
        boolean_features = X_train.select_dtypes(include=["bool"]).columns

        # Pipeline appliqué aux variables numériques :
        # imputation des valeurs manquantes puis mise à l'échelle
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.numeric_imputer_strategy)),
                ("scaler", self.construire_scaler()),
            ]
        )

        # Pipeline appliqué aux variables catégorielles :
        # imputation puis encodage one-hot
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.categorical_imputer_strategy)),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Regroupement des traitements par type de variable
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
                ("bool", "passthrough", boolean_features),
            ]
        )

        # Pipeline final : prétraitement des données puis entraînement du modèle
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", self.construire_model()),
            ]
        )

        return self.pipeline