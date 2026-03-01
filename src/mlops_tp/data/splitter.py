from sklearn.model_selection import train_test_split
from mlops_tp.config import RANDOM_STATE
from mlops_tp.data.loader import load_train_data, load_test_data

# =========================
# SPLIT TRAIN / VALIDATION
# =========================
def split_train_val(load_train_data):
    X = load_train_data.drop(columns=["Priority_Label"])
    y = load_train_data["Priority_Label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=RANDOM_STATE
    )

    return X_train, X_val, y_train, y_val


# =========================
# SPLIT TEST
# =========================
def split_test(load_test_data):
    X_test = load_test_data.drop(columns=["Priority_Label"])
    y_test = load_test_data["Priority_Label"]

    return X_test, y_test