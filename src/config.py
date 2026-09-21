from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"
MODEL_DIR = BASE_DIR / "models"

for directory in [
    PROCESSED_DIR,
    OUTPUT_DIR,
    MODEL_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# RAW FILES
# ============================================================

ORDERS_PATH = RAW_DIR / "orders.csv"
PRIOR_PATH = RAW_DIR / "order_products__prior.csv"
TRAIN_PATH = RAW_DIR / "order_products__train.csv"
PRODUCTS_PATH = RAW_DIR / "products.csv"
AISLES_PATH = RAW_DIR / "aisles.csv"
DEPARTMENTS_PATH = RAW_DIR / "departments.csv"


# ============================================================
# MODELING
# ============================================================

RANDOM_STATE = 42

TARGET_COL = "target"

USER_COL = "user_id"
PRODUCT_COL = "product_id"


# ============================================================
# CANDIDATE GENERATION
# ============================================================
TOP_GLOBAL_PRODUCTS = 30
TOP_PRODUCTS_PER_DEPARTMENT = 5
MAX_HISTORICAL_PRODUCTS_PER_USER = 50

# ============================================================
# NEGATIVE SAMPLING
# ============================================================

NEGATIVE_RATIO = 3


# ============================================================
# TRAIN / VALIDATION
# ============================================================

VALIDATION_QUANTILE = 0.75


# ============================================================
# SCORING
# ============================================================

TOP_N_RECOMMENDATIONS = 10