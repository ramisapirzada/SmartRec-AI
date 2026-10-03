from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "src" / "models"

# Files
DATASET_PATH = DATA_DIR / "learning_resources.csv"

TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
ITEM_VECTORS_PATH = MODEL_DIR / "item_vectors.joblib"
CATALOG_PATH = MODEL_DIR / "resource_catalog.pkl"