import pandas as pd
from mlops_tp.config import RAW_DATA_DIR

def load_train_data():
    return pd.read_csv(RAW_DATA_DIR / "university_query_train.csv")

def load_test_data():
    return pd.read_csv(RAW_DATA_DIR / "university_query_test.csv")