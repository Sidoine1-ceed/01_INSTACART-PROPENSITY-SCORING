import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    PROCESSED_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    VALIDATION_QUANTILE
)


# ============================================================
# 1. LOAD TRAINING DATASET
# ============================================================

print("=" * 60)
print("LOADING TRAINING DATASET")
print("=" * 60)

df = pd.read_parquet(
    PROCESSED_DIR / "training_dataset.parquet"
)

print("Dataset:", df.shape)


# ============================================================
# 2. CHECK SPLIT VARIABLE
# ============================================================

print("\n" + "=" * 60)
print("CHECKING SPLIT VARIABLE")
print("=" * 60)

print(
    "customer_total_orders - NaN:",
    df["customer_total_orders"].isna().sum()
)

print(
    "customer_total_orders - non NaN:",
    df["customer_total_orders"].notna().sum()
)

print(
    "Min:",
    df["customer_total_orders"].min()
)

print(
    "Max:",
    df["customer_total_orders"].max()
)


# ============================================================
# 3. TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / VALIDATION SPLIT")
print("=" * 60)

split_value = (
    df["customer_total_orders"]
    .quantile(VALIDATION_QUANTILE)
)

print(
    "Split value:",
    split_value
)

train_df = (
    df[
        df["customer_total_orders"] <= split_value
    ]
    .copy()
)

valid_df = (
    df[
        df["customer_total_orders"] > split_value
    ]
    .copy()
)

print(
    "Train:",
    train_df.shape
)

print(
    "Validation:",
    valid_df.shape
)

print(
    "Total used:",
    len(train_df) + len(valid_df)
)

print(
    "Total dataset:",
    len(df)
)


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print("\nTrain:")
print(
    train_df["target"]
    .value_counts(normalize=True)
)

print("\nValidation:")
print(
    valid_df["target"]
    .value_counts(normalize=True)
)


# ============================================================
# 5. DEFINE FEATURES
# ============================================================

DROP_COLUMNS = [
    "user_id",
    "product_id",
    "target",
    "first_order_number",
    "last_order_number"
]

FEATURES = [
    col
    for col in df.columns
    if col not in DROP_COLUMNS
]

print(
    "\nNumber of features:",
    len(FEATURES)
)

print("\nFeatures:")

for feature in FEATURES:
    print("-", feature)


# ============================================================
# 6. PREPARE DATA
# ============================================================

print("\n" + "=" * 60)
print("PREPARING DATA")
print("=" * 60)

X_train = (
    train_df[FEATURES]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)

X_valid = (
    valid_df[FEATURES]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)

y_train = (
    train_df["target"]
    .astype("int8")
)

y_valid = (
    valid_df["target"]
    .astype("int8")
)

X_train = X_train.astype(
    np.float32
)

X_valid = X_valid.astype(
    np.float32
)

print("\nX_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("y_train:", y_train.shape)
print("y_valid:", y_valid.shape)


# ============================================================
# 7. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

logistic_model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=200,
                solver="lbfgs",
                random_state=RANDOM_STATE
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_path = (
    MODEL_DIR
    / "logistic_model.joblib"
)

joblib.dump(
    logistic_model,
    logistic_path
)

print(
    "\nLogistic Regression saved to:"
)

print(logistic_path)


# ============================================================
# 8. RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=20,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_path = (
    MODEL_DIR
    / "random_forest_model.joblib"
)

joblib.dump(
    random_forest_model,
    random_forest_path
)

print(
    "\nRandom Forest saved to:"
)

print(random_forest_path)


# ============================================================
# 9. SAVE FEATURE LIST
# ============================================================

features_path = (
    MODEL_DIR
    / "features.joblib"
)

joblib.dump(
    FEATURES,
    features_path
)

print(
    "\nFeature list saved to:"
)

print(features_path)


# ============================================================
# 10. SAVE SPLIT INFORMATION
# ============================================================

split_info = {
    "split_variable": "customer_total_orders",
    "split_value": float(split_value),
    "validation_quantile": VALIDATION_QUANTILE,
    "train_size": len(train_df),
    "validation_size": len(valid_df)
}

joblib.dump(
    split_info,
    MODEL_DIR / "split_info.joblib"
)


# ============================================================
# 11. TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nModels created:")
print("1. Logistic Regression")
print("2. Random Forest")

print("\nSaved files:")
print("- logistic_model.joblib")
print("- random_forest_model.joblib")
print("- features.joblib")
print("- split_info.joblib")

print("\nLightGBM is NOT used.")