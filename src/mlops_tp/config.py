from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR

ARTIFACT_DIR = BASE_DIR / "src" / "mlops_tp" / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
FEATURE_SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"
RUN_INFO_PATH = ARTIFACT_DIR / "run_info.json"

RANDOM_STATE = 42
MODEL_VERSION = "0.1.0"