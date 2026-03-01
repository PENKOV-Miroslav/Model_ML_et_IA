from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from mlops_tp.train import Train


class PipelineModel:

    def __init__(self):
        self.trainer = Train()
        self.pipeline = None

    def creer_pipeline(self, X_train):
        # Identifier les colonnes numériques et catégorielles
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns

        # Créer les transformations pour les colonnes numériques et catégorielles
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Combiner les transformations dans un ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Créer la pipeline complète avec un classifieur
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=self.trainer.random_state))
        ])