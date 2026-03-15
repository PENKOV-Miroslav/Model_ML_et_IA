from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from mlops_tp.train import Train


class PipelineModel:

    def __init__(self):
        self.trainer = Train()
        self.pipeline = None

    def creer_pipeline(self, X_train):

        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns

        # Pipeline numérique
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        # Pipeline catégoriel
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(random_state=self.trainer.random_state))
            ]
        )
        return self.pipeline